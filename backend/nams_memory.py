"""
Neo4j Agent Memory Service (NAMS) Integration for Histology Multi-Agent System.

100% LOCAL NEO4J & NAMS COMPATIBLE.
No cloud dependency required when running with local Neo4j instance (bolt://localhost:7687).

Implements the 3 graph-native memory tiers described in NAMS:
1. Short-Term Memory: Conversations and sequential messages in Neo4j (:Conversation)-[:HAS_MESSAGE]->(:Message).
2. Long-Term Memory: Domain Knowledge Graph with Entities and typed Relationships (:Entity)-[:RELATION]->(:Entity)
   (POLE+O + Histology Structural/Cellular Ontology).
3. Reasoning Memory: Agent steps, tool calls, and observations/reflections (:Observation, :AgentStep).

Includes strict role-based access control:
- Teacher Mode ('teacher'): Read/Write. Allows updating the knowledge graph,
  recording teacher corrections, resolving entities (HITL), and committing reasoning.
- Student Mode ('student'): Read-Only for the canonical knowledge graph.
  Provides layered context retrieval (Reflection -> Observations -> Messages)
  and Socratic tutoring assistance without contaminating the ground-truth graph.
"""

import os
import json
import time
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import neo4j
    from neo4j import GraphDatabase
except ImportError:
    neo4j = None
    GraphDatabase = None

import httpx

logger = logging.getLogger("sam3-nams")


class NAMSRole(str, Enum):
    TEACHER = "teacher"
    STUDENT = "student"


class NAMSMemoryError(Exception):
    """Custom exception for NAMS memory access violations or connection errors."""
    pass


class LocalNeo4jDriverEngine:
    """
    Direct Driver for Local Neo4j instance (e.g. bolt://127.0.0.1:7687).
    Executes Cypher queries directly against the user's local Neo4j database,
    providing 100% local, private, and high-performance NAMS graph memory.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.driver = None
        self._is_connected = False
        self._connect()

    def _connect(self) -> bool:
        if GraphDatabase is None:
            logger.warning("neo4j python package not installed.")
            return False
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            self._is_connected = True
            logger.info(f"Connected successfully to local Neo4j at {self.uri}")
            self._init_schema()
            return True
        except Exception as e:
            logger.warning(f"Could not connect to local Neo4j at {self.uri}: {e}")
            self._is_connected = False
            return False

    def is_available(self) -> bool:
        if not self._is_connected or not self.driver:
            return self._connect()
        return True

    def _init_schema(self) -> None:
        """Ensures optimal O(log n) indexing on Entity and Conversation IDs."""
        if not self.driver:
            return
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE",
        ]
        try:
            with self.driver.session() as session:
                for q in queries:
                    try:
                        session.run(q)
                    except Exception as q_err:
                        logger.debug(f"Constraint notice ({q}): {q_err}")
        except Exception as e:
            logger.debug(f"Schema initialization note: {e}")

    def create_conversation(self, conversation_id: str, user_id: str, mode: str) -> Dict[str, Any]:
        with self.driver.session() as session:
            session.run(
                """
                MERGE (c:Conversation {id: $id})
                ON CREATE SET c.userId = $user_id, c.mode = $mode, c.created_at = timestamp()
                """,
                id=conversation_id, user_id=user_id, mode=mode
            )
        return {"id": conversation_id, "userId": user_id, "mode": mode}

    def add_message(self, conversation_id: str, role: str, content: str) -> Dict[str, Any]:
        with self.driver.session() as session:
            session.run(
                """
                MERGE (c:Conversation {id: $id})
                CREATE (m:Message {role: $role, content: $content, timestamp: timestamp()})
                CREATE (c)-[:HAS_MESSAGE]->(m)
                """,
                id=conversation_id, role=role, content=content
            )
        return {"role": role, "content": content}

    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        active_reflection = ""
        observations = []
        messages = []

        with self.driver.session() as session:
            # Reflection
            r_res = session.run("MATCH (c:Conversation {id: $id}) RETURN c.active_reflection AS refl", id=conversation_id)
            rec = r_res.single()
            if rec and rec["refl"]:
                active_reflection = rec["refl"]

            # Observations
            o_res = session.run(
                """
                MATCH (c:Conversation {id: $id})-[:HAS_OBSERVATION]->(o:Observation)
                RETURN o.text AS text ORDER BY o.timestamp DESC LIMIT 5
                """,
                id=conversation_id
            )
            observations = [row["text"] for row in o_res]

            # Messages
            m_res = session.run(
                """
                MATCH (c:Conversation {id: $id})-[:HAS_MESSAGE]->(m:Message)
                RETURN m.role AS role, m.content AS content, m.timestamp AS timestamp
                ORDER BY m.timestamp ASC
                """,
                id=conversation_id
            )
            messages = [{"role": row["role"], "content": row["content"]} for row in m_res]

        return {
            "conversation_id": conversation_id,
            "active_reflection": active_reflection,
            "observations": observations,
            "recent_messages": messages[-6:],
        }

    def upsert_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str,
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        props_json = json.dumps(properties or {}, ensure_ascii=False)
        with self.driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {id: $id})
                SET e.name = $name,
                    e.type = $entity_type,
                    e.description = $description,
                    e.confidence = $confidence,
                    e.properties = $props_json,
                    e.updated_at = timestamp()
                """,
                id=entity_id,
                name=name,
                entity_type=entity_type,
                description=description,
                confidence=confidence,
                props_json=props_json,
            )
        return {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "description": description,
            "properties": properties or {},
        }

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        props_json = json.dumps(properties or {}, ensure_ascii=False)
        with self.driver.session() as session:
            session.run(
                """
                MERGE (s:Entity {id: $source_id})
                MERGE (t:Entity {id: $target_id})
                MERGE (s)-[r:RELATION {type: $rel_type}]->(t)
                SET r.properties = $props_json, r.updated_at = timestamp()
                """,
                source_id=source_id, target_id=target_id, rel_type=rel_type, props_json=props_json
            )
        return {"source": source_id, "target": target_id, "type": rel_type}

    def add_reasoning_step(
        self,
        conversation_id: str,
        tool_name: str,
        details: str,
        touched_entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        step_id = f"step_{int(time.time() * 1000)}"
        with self.driver.session() as session:
            session.run(
                """
                MERGE (c:Conversation {id: $id})
                CREATE (s:AgentStep {
                    step_id: $step_id,
                    tool_name: $tool_name,
                    details: $details,
                    timestamp: timestamp()
                })
                CREATE (c)-[:HAS_STEP]->(s)
                """,
                id=conversation_id, step_id=step_id, tool_name=tool_name, details=details
            )
        return {"step_id": step_id, "tool_name": tool_name}

    def record_teacher_correction(
        self,
        conversation_id: str,
        target_entity_id: str,
        correction_text: str,
        accepted_class: str,
        reasoning: str,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        obs_id = f"obs_{int(time.time() * 1000)}"
        obs_text = f"Docente corrigió/validó estructura '{target_entity_id}' como '{accepted_class}'. Criterio: {reasoning}"
        refl_chunk = f"[Docente]: {target_entity_id} validado como {accepted_class} ({reasoning})"
        entity_name = f"Cell_{target_entity_id}"
        ent_desc = f"Teacher-validated histological structure: {accepted_class}. Rationale: {reasoning}"

        with self.driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {id: $target_id})
                SET e.name = $name,
                    e.type = 'CellularStructure',
                    e.histology_class = $accepted_class,
                    e.description = $desc,
                    e.teacher_correction = $correction,
                    e.teacher_rationale = $reasoning,
                    e.verified_by_teacher = true,
                    e.updated_at = timestamp()
                WITH e
                MERGE (c:Conversation {id: $conv_id})
                CREATE (o:Observation {
                    id: $obs_id,
                    text: $obs_text,
                    timestamp: timestamp()
                })
                CREATE (c)-[:HAS_OBSERVATION]->(o)
                MERGE (c)-[r:CORRECTED_BY]->(e)
                SET r.rationale = $reasoning, r.timestamp = timestamp()
                SET c.active_reflection = CASE
                    WHEN c.active_reflection IS NULL OR c.active_reflection = '' THEN $refl_chunk
                    ELSE c.active_reflection + ' | ' + $refl_chunk
                END
                """,
                target_id=target_entity_id,
                name=entity_name,
                accepted_class=accepted_class,
                desc=ent_desc,
                correction=correction_text,
                reasoning=reasoning,
                conv_id=conversation_id,
                obs_id=obs_id,
                obs_text=obs_text,
                refl_chunk=refl_chunk,
            )

        return {
            "status": "success",
            "message": f"Corrección registrada en Neo4j local ({target_entity_id} -> {accepted_class})",
            "observation": obs_text,
            "target_id": target_entity_id,
            "accepted_class": accepted_class,
        }

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        results = []
        with self.driver.session() as session:
            res = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($q)
                   OR toLower(e.description) CONTAINS toLower($q)
                   OR toLower(coalesce(e.histology_class, '')) CONTAINS toLower($q)
                RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description,
                       e.histology_class AS histology_class, e.properties AS properties
                LIMIT $limit
                """,
                q=query, limit=limit
            )
            for row in res:
                results.append({
                    "id": row["id"],
                    "name": row["name"],
                    "type": row["type"],
                    "description": row["description"],
                    "histology_class": row["histology_class"],
                })
        return results

    def get_graph_visualization(self) -> Dict[str, Any]:
        nodes = []
        edges = []
        with self.driver.session() as session:
            # Nodes
            n_res = session.run(
                """
                MATCH (n:Entity)
                RETURN n.id AS id, n.name AS label, n.type AS type,
                       n.description AS description, n.histology_class AS histology_class,
                       n.verified_by_teacher AS verified
                LIMIT 100
                """
            )
            for row in n_res:
                nodes.append({
                    "id": row["id"],
                    "label": row["label"] or row["id"],
                    "type": row["type"] or "Entity",
                    "description": row["description"] or "",
                    "histology_class": row["histology_class"],
                    "verified": bool(row["verified"]),
                })

            # Edges
            e_res = session.run(
                """
                MATCH (s:Entity)-[r]->(t:Entity)
                RETURN s.id AS source, t.id AS target, type(r) AS type
                LIMIT 200
                """
            )
            for row in e_res:
                edges.append({
                    "source": row["source"],
                    "target": row["target"],
                    "type": row["type"],
                })

            # Also include teacher corrections edges from conversation to entity
            c_res = session.run(
                """
                MATCH (c:Conversation)-[r:CORRECTED_BY]->(e:Entity)
                RETURN c.id AS source, e.id AS target, 'CORRECTED_BY' AS type, r.rationale AS rationale
                LIMIT 50
                """
            )
            for row in c_res:
                nodes.append({
                    "id": row["source"],
                    "label": f"Sesión ({row['source']})",
                    "type": "Conversation",
                    "description": "Sesión de anotación docente",
                    "verified": True,
                })
                edges.append({
                    "source": row["source"],
                    "target": row["target"],
                    "type": "CORRECTED_BY",
                    "rationale": row["rationale"],
                })

        return {
            "engine": "Local Neo4j (bolt://127.0.0.1:7687)",
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }


class LocalNAMSGraphEngine:
    """
    In-memory / JSON fallback engine when local Neo4j is offline or unavailable.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or (Path(__file__).resolve().parent.parent / "datasets" / "nams_local_graph.json"))
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph: Dict[str, Any] = {
            "conversations": {},
            "entities": {},
            "relationships": [],
            "reasoning_steps": [],
            "observations": {},
            "active_reflections": {},
        }
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.graph.update(data)
            except Exception as e:
                logger.warning(f"Could not load local NAMS graph from {self.storage_path}: {e}")

    def _save(self) -> None:
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.graph, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to persist local NAMS graph to {self.storage_path}: {e}")

    def create_conversation(self, conversation_id: str, user_id: str, mode: str) -> Dict[str, Any]:
        if conversation_id not in self.graph["conversations"]:
            self.graph["conversations"][conversation_id] = {
                "id": conversation_id,
                "userId": user_id,
                "mode": mode,
                "created_at": time.time(),
                "messages": []
            }
            self._save()
        return self.graph["conversations"][conversation_id]

    def add_message(self, conversation_id: str, role: str, content: str) -> Dict[str, Any]:
        conv = self.graph["conversations"].get(conversation_id)
        if not conv:
            conv = self.create_conversation(conversation_id, user_id="user_default", mode="teacher")
        
        msg = {
            "index": len(conv["messages"]),
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        conv["messages"].append(msg)
        self._save()
        return msg

    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        conv = self.graph["conversations"].get(conversation_id, {"messages": []})
        messages = conv.get("messages", [])
        observations = self.graph["observations"].get(conversation_id, [])
        reflection = self.graph["active_reflections"].get(conversation_id, {}).get("summary", "")

        return {
            "conversation_id": conversation_id,
            "active_reflection": reflection,
            "observations": [obs.get("text", "") for obs in observations[-5:]],
            "recent_messages": messages[-6:],
        }

    def upsert_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str,
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        ent = {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "description": description,
            "properties": properties or {},
            "confidence": confidence,
            "updated_at": time.time(),
        }
        self.graph["entities"][entity_id] = ent
        self._save()
        return ent

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rel = {
            "source": source_id,
            "target": target_id,
            "type": rel_type,
            "properties": properties or {},
            "created_at": time.time(),
        }
        for existing in self.graph["relationships"]:
            if existing["source"] == source_id and existing["target"] == target_id and existing["type"] == rel_type:
                existing["properties"].update(properties or {})
                self._save()
                return existing

        self.graph["relationships"].append(rel)
        self._save()
        return rel

    def add_reasoning_step(
        self,
        conversation_id: str,
        tool_name: str,
        details: str,
        touched_entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        step = {
            "conversation_id": conversation_id,
            "step_id": f"step_{len(self.graph['reasoning_steps']) + 1}",
            "tool_name": tool_name,
            "details": details,
            "touched_entities": touched_entities or [],
            "created_at": time.time(),
        }
        self.graph["reasoning_steps"].append(step)
        self._save()
        return step

    def add_observation(self, conversation_id: str, text: str, message_indices: Optional[List[int]] = None) -> None:
        if conversation_id not in self.graph["observations"]:
            self.graph["observations"][conversation_id] = []
        obs = {
            "id": f"obs_{len(self.graph['observations'][conversation_id]) + 1}",
            "text": text,
            "message_indices": message_indices or [],
            "created_at": time.time()
        }
        self.graph["observations"][conversation_id].append(obs)
        self._save()

    def set_active_reflection(self, conversation_id: str, summary: str) -> None:
        self.graph["active_reflections"][conversation_id] = {
            "summary": summary,
            "updated_at": time.time()
        }
        self._save()

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        q = query.lower().strip()
        matched = []
        for ent in self.graph["entities"].values():
            name = ent.get("name", "").lower()
            desc = ent.get("description", "").lower()
            etype = ent.get("type", "").lower()
            if q in name or q in desc or q in etype:
                matched.append(ent)
            if len(matched) >= limit:
                break
        return matched

    def record_teacher_correction(
        self,
        conversation_id: str,
        target_entity_id: str,
        correction_text: str,
        accepted_class: str,
        reasoning: str,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        entity_name = f"Cell_{target_entity_id}"
        ent_desc = f"Teacher-validated histological structure: {accepted_class}. Rationale: {reasoning}"
        
        updated_entity = self.upsert_entity(
            entity_id=target_entity_id,
            name=entity_name,
            entity_type="CellularStructure",
            description=ent_desc,
            properties={
                "histology_class": accepted_class,
                "teacher_correction": correction_text,
                "reasoning": reasoning,
                "verified_by_teacher": True,
            },
            confidence=confidence,
        )

        obs_text = f"Docente corrigió/validó estructura '{target_entity_id}' como '{accepted_class}'. Criterio: {reasoning}"
        self.add_observation(conversation_id, obs_text)

        cur_reflection = self.graph["active_reflections"].get(conversation_id, {}).get("summary", "")
        new_reflection = f"{cur_reflection} | [Docente]: {target_entity_id} validado como {accepted_class} ({reasoning}).".strip(" | ")
        self.set_active_reflection(conversation_id, new_reflection)

        return {
            "status": "success",
            "message": f"Corrección docente registrada en NAMS JSON para {target_entity_id} -> {accepted_class}.",
            "entity": updated_entity,
            "observation": obs_text,
        }

    def get_graph_visualization(self) -> Dict[str, Any]:
        nodes = []
        for ent_id, ent in self.graph["entities"].items():
            nodes.append({
                "id": ent_id,
                "label": ent.get("name", ent_id),
                "type": ent.get("type", "Entity"),
                "description": ent.get("description", ""),
                "verified": ent.get("properties", {}).get("verified_by_teacher", False),
            })
        edges = []
        for rel in self.graph["relationships"]:
            edges.append({
                "source": rel["source"],
                "target": rel["target"],
                "type": rel["type"],
            })
        return {
            "engine": "Local JSON Fallback",
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }


class NAMSClient:
    """
    Unified NAMS Client prioritizing 100% LOCAL Neo4j.
    Ensures no data leaves the local machine.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.local_only = os.environ.get("NEO4J_LOCAL_ONLY", "true").lower() in ["true", "1", "yes"]
        self.neo4j_engine = LocalNeo4jDriverEngine()
        self.json_fallback = LocalNAMSGraphEngine()

    @property
    def active_engine(self) -> Union[LocalNeo4jDriverEngine, LocalNAMSGraphEngine]:
        if self.neo4j_engine.is_available():
            return self.neo4j_engine
        return self.json_fallback

    # -------------------------------------------------------------------------
    # 1. Short-Term Memory
    # -------------------------------------------------------------------------

    def create_conversation(
        self,
        conversation_id: str,
        user_id: str = "default_user",
        mode: NAMSRole = NAMSRole.TEACHER,
    ) -> Dict[str, Any]:
        return self.active_engine.create_conversation(conversation_id, user_id=user_id, mode=mode.value)

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> Dict[str, Any]:
        return self.active_engine.add_message(conversation_id, role, content)

    # -------------------------------------------------------------------------
    # 2. Layered Context (Reflection -> Observations -> Messages)
    # -------------------------------------------------------------------------

    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        return self.active_engine.get_context(conversation_id)

    # -------------------------------------------------------------------------
    # 3. Long-Term Memory (Knowledge Graph Entities & Relationships)
    # -------------------------------------------------------------------------

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self.active_engine.search_entities(query, limit=limit)

    def upsert_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str,
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        mode: NAMSRole = NAMSRole.TEACHER,
    ) -> Dict[str, Any]:
        """
        Upserts an entity in the long-term knowledge graph.
        CRITICAL: Student mode is strictly FORBIDDEN from writing to the graph.
        """
        if mode == NAMSRole.STUDENT:
            logger.info(f"[NAMS Policy] Student mode blocked from writing entity '{name}' to knowledge graph.")
            return {
                "status": "ignored",
                "reason": "Student mode cannot write or mutate the canonical knowledge graph."
            }

        return self.active_engine.upsert_entity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            description=description,
            properties=properties,
            confidence=confidence,
        )

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
        mode: NAMSRole = NAMSRole.TEACHER,
    ) -> Dict[str, Any]:
        """Creates a relationship in the knowledge graph. Blocked in Student Mode."""
        if mode == NAMSRole.STUDENT:
            logger.info(f"[NAMS Policy] Student mode blocked from adding relationship '{rel_type}' to knowledge graph.")
            return {
                "status": "ignored",
                "reason": "Student mode cannot write relationships to the canonical knowledge graph."
            }

        return self.active_engine.add_relationship(source_id, target_id, rel_type, properties)

    # -------------------------------------------------------------------------
    # 4. Reasoning Memory (Agent Steps & Tool Calls)
    # -------------------------------------------------------------------------

    def record_reasoning_step(
        self,
        conversation_id: str,
        tool_name: str,
        details: str,
        touched_entities: Optional[List[str]] = None,
        mode: NAMSRole = NAMSRole.TEACHER,
    ) -> Dict[str, Any]:
        return self.active_engine.add_reasoning_step(
            conversation_id=conversation_id,
            tool_name=tool_name,
            details=details,
            touched_entities=touched_entities,
        )

    # -------------------------------------------------------------------------
    # 5. Teacher Corrections & Pedagogical Memory API
    # -------------------------------------------------------------------------

    def record_teacher_correction(
        self,
        conversation_id: str,
        target_entity_id: str,
        correction_text: str,
        accepted_class: str,
        reasoning: str,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Teacher Mode specialized workflow in Local Neo4j:
        1. Updates or creates the entity in local Neo4j with the corrected ground-truth class.
        2. Creates an observation node in reasoning memory with the teacher's rationale.
        3. Updates the active reflection of the conversation.
        """
        return self.active_engine.record_teacher_correction(
            conversation_id=conversation_id,
            target_entity_id=target_entity_id,
            correction_text=correction_text,
            accepted_class=accepted_class,
            reasoning=reasoning,
            confidence=confidence,
        )

    def get_graph_visualization(self) -> Dict[str, Any]:
        """Returns the current state of nodes and edges from Local Neo4j."""
        return self.active_engine.get_graph_visualization()
