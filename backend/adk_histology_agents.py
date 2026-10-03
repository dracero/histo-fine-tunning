"""
Specialized ADK / A2A Multi-Agent System for Histological Analysis with NAMS Memory.

Architecture:
1. SpatialOntologyAgent: Computes geometric containment, topological hierarchies,
   and compartment inclusions (e.g. Lumen, Visceral Layer, Cortex/Medulla).
2. TextualOntologyAgent: Queries domain knowledge bases, synonyms, POLE+O criteria,
   and cellular taxonomies (e.g. SNOMED, Uberon, JSON ontologies).
3. ImageAnalysisAgent: Multimodal vision reasoning using Gemini 3.5 Flash & SAM3/Virchow
   under torch.inference_mode().
4. NAMSMemoryAgent: Gatekeeper for Neo4j Agent Memory Service (NAMS).
   Enforces Teacher Mode (read/write, HITL corrections, reasoning graph audit)
   vs Student Mode (read-only, pedagogical context injection, graph protection).
5. PedagogicalOrchestratorAgent: A2A-compliant coordinator dispatching requests
   across specialist agents and synthesizing technical or Socratic responses.
"""

import os
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

import cv2
import numpy as np

try:
    from backend.nams_memory import NAMSClient, NAMSRole
    from backend.gemini_vision import generate_gemini_content, GEMINI_MODEL
    from backend.pdf_ontology import load_ontology, list_ontologies
except ImportError:
    from nams_memory import NAMSClient, NAMSRole
    from gemini_vision import generate_gemini_content, GEMINI_MODEL
    from pdf_ontology import load_ontology, list_ontologies

logger = logging.getLogger("sam3-adk-agents")


# =============================================================================
# Agent 1: Spatial Ontology Agent
# =============================================================================

class SpatialOntologyAgent:
    """Specialist in geometric topology, spatial hierarchies, and compartment containment."""

    def __init__(self):
        self.name = "SpatialOntologyAgent"

    @staticmethod
    def _is_point_enclosed(pt: Tuple[float, float], polys: List[np.ndarray]) -> bool:
        """Determines if a point is inside any polygon of a structure."""
        x, y = float(pt[0]), float(pt[1])
        for poly in polys:
            try:
                if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                    return True
            except Exception as e:
                logger.warning(f"Error in spatial polygon test: {e}")
        return False

    def analyze_spatial_relationships(
        self,
        cell_points: List[Tuple[float, float]],
        macro_polygons: Dict[str, List[np.ndarray]],
        image_size: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """
        Determines topological containment of cells inside macro structures
        using deterministic polygon point testing (O(log n) ray-casting).
        """
        results: List[Dict[str, Any]] = []
        for idx, pt in enumerate(cell_points):
            enclosing_structures = [
                struct_name
                for struct_name, polys in macro_polygons.items()
                if self._is_point_enclosed(pt, polys)
            ]
            results.append({
                "cell_index": idx,
                "point": pt,
                "enclosing_structures": enclosing_structures,
                "spatial_context": (
                    f"Inside {', '.join(enclosing_structures)}" if enclosing_structures
                    else "Interstitium / Stroma"
                )
            })
        return results


# =============================================================================
# Agent 2: Textual Ontology Agent
# =============================================================================

class TextualOntologyAgent:
    """Specialist in biological taxonomy, cytology criteria, and domain literature."""

    def __init__(self):
        self.name = "TextualOntologyAgent"
        self._cached_ontologies: Dict[str, Dict[str, Any]] = {}

    def get_ontology_definition(self, ontology_name: str, target_class: Optional[str] = None) -> Dict[str, Any]:
        """Loads and indexes ontology data with dictionary lookups (O(1))."""
        if ontology_name not in self._cached_ontologies:
            try:
                ont_data = load_ontology(ontology_name)
                # Build fast index by class name
                indexed = {
                    item.get("name", "").lower(): item
                    for item in ont_data.get("structures", [])
                    if isinstance(item, dict) and "name" in item
                }
                self._cached_ontologies[ontology_name] = {
                    "raw": ont_data,
                    "indexed": indexed,
                }
            except Exception as e:
                logger.error(f"Failed to load textual ontology '{ontology_name}': {e}")
                return {"error": f"Ontology {ontology_name} not found"}

        ont_entry = self._cached_ontologies[ontology_name]
        if target_class:
            match = ont_entry["indexed"].get(target_class.lower().strip())
            return match or {"error": f"Class '{target_class}' not found in {ontology_name}"}

        return ont_entry["raw"]


# =============================================================================
# Agent 3: Multimodal Image Analysis Agent
# =============================================================================

class ImageAnalysisAgent:
    """Specialist in visual feature extraction and morphological inspection using Gemini Vision."""

    def __init__(self):
        self.name = "ImageAnalysisAgent"

    def describe_morphology(
        self,
        image: Image.Image,
        region_description: str,
        spatial_hints: Optional[str] = None,
    ) -> str:
        """Runs multimodal vision inspection on histological image."""
        system_instruction = (
            "Eres un patólogo histológico de alta precisión. Describe las características "
            "citológicas visibles: tamaño nuclear, cromatina, relación núcleo-citoplasma, "
            "forma celular (cúbica, cilíndrica, plana), bordes apicales y tinción tintorial (H&E)."
        )
        prompt = f"Analiza histológicamente la región: {region_description}."
        if spatial_hints:
            prompt += f"\nContexto espacial del tejido: {spatial_hints}."

        try:
            response = generate_gemini_content(
                contents=[image, prompt],
                system_instruction=system_instruction,
                temperature=0.2,
                preferred_model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            logger.error(f"ImageAnalysisAgent failed: {e}")
            return f"Error en análisis visual de la imagen: {e}"


# =============================================================================
# Agent 4: NAMS Memory Agent (Gatekeeper & Context Layer)
# =============================================================================

class NAMSMemoryAgent:
    """
    Gatekeeper agent for Neo4j Agent Memory Service.
    Enforces role-based permissions (Teacher: Read/Write vs Student: Read-Only).
    """

    def __init__(self, nams_client: Optional[NAMSClient] = None):
        self.name = "NAMSMemoryAgent"
        self.client = nams_client or NAMSClient()

    def get_layered_context(self, conversation_id: str) -> Dict[str, Any]:
        """Fetches active reflection, observations, and recent messages."""
        return self.client.get_context(conversation_id)

    def process_correction_or_query(
        self,
        conversation_id: str,
        user_id: str,
        mode: NAMSRole,
        user_text: str,
        target_entity_id: Optional[str] = None,
        accepted_class: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes memory events strictly according to role:
        - TEACHER: records correction, updates long-term knowledge graph, updates reflections.
        - STUDENT: reads context, refuses graph writes, returns context for tutor.
        """
        # Ensure conversation exists
        self.client.create_conversation(conversation_id, user_id=user_id, mode=mode)
        self.client.add_message(conversation_id, role="user", content=user_text)

        if mode == NAMSRole.TEACHER and target_entity_id and accepted_class:
            # Teacher writes ground-truth into NAMS
            result = self.client.record_teacher_correction(
                conversation_id=conversation_id,
                target_entity_id=target_entity_id,
                correction_text=user_text,
                accepted_class=accepted_class,
                reasoning=reasoning or "Validación por docente",
            )
            return {
                "role": mode.value,
                "action": "knowledge_graph_updated",
                "details": result,
                "context": self.client.get_context(conversation_id),
            }

        # Student mode: Read-only context retrieval
        context = self.client.get_context(conversation_id)
        return {
            "role": mode.value,
            "action": "context_retrieved",
            "context": context,
            "can_write": False,
        }


# =============================================================================
# Agent 5: Pedagogical Orchestrator Agent (A2A Coordinator)
# =============================================================================

class PedagogicalOrchestratorAgent:
    """
    A2A Coordinator orchestrating the 4 specialized agents.
    Provides Teacher Mode (technical analysis + HITL graph commits)
    and Student Mode (Socratic tutoring + graph protection).
    """

    def __init__(self, nams_client: Optional[NAMSClient] = None):
        self.spatial_agent = SpatialOntologyAgent()
        self.textual_agent = TextualOntologyAgent()
        self.image_agent = ImageAnalysisAgent()
        self.memory_agent = NAMSMemoryAgent(nams_client=nams_client)

    def run_cycle(
        self,
        image: Optional[Image.Image],
        user_query: str,
        mode: NAMSRole = NAMSRole.TEACHER,
        conversation_id: str = "default_session",
        user_id: str = "docente_01",
        ontology_name: str = "arch4",
        macro_polygons: Optional[Dict[str, List[np.ndarray]]] = None,
        cell_points: Optional[List[Tuple[float, float]]] = None,
        target_entity_id: Optional[str] = None,
        accepted_class: Optional[str] = None,
        teacher_rationale: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes a complete A2A orchestration cycle.
        """
        start_t = time.time()
        agent_steps: List[str] = []

        # 1. Memory Context
        memory_result = self.memory_agent.process_correction_or_query(
            conversation_id=conversation_id,
            user_id=user_id,
            mode=mode,
            user_text=user_query,
            target_entity_id=target_entity_id,
            accepted_class=accepted_class,
            reasoning=teacher_rationale,
        )
        context = memory_result.get("context", {})
        agent_steps.append("NAMSMemoryAgent: Contexto estratificado recuperado")

        # 2. Spatial Analysis if polygons & points exist
        spatial_info = []
        if macro_polygons and cell_points and image:
            spatial_info = self.spatial_agent.analyze_spatial_relationships(
                cell_points=cell_points,
                macro_polygons=macro_polygons,
                image_size=image.size,
            )
            agent_steps.append(f"SpatialOntologyAgent: Analizadas {len(spatial_info)} relaciones topológicas")

        # 3. Textual Ontology Information
        ont_info = self.textual_agent.get_ontology_definition(ontology_name, accepted_class)
        agent_steps.append(f"TextualOntologyAgent: Ontología '{ontology_name}' consultada")

        # 4. Multimodal Image Analysis
        image_desc = ""
        if image:
            spatial_summary = ", ".join([f"Célula {s['cell_index']}: {s['spatial_context']}" for s in spatial_info[:3]])
            image_desc = self.image_agent.describe_morphology(
                image=image,
                region_description=user_query,
                spatial_hints=spatial_summary,
            )
            agent_steps.append("ImageAnalysisAgent: Descripción morfológica extraída")

        # 5. Final Synthesis: Teacher vs Student Mode
        if mode == NAMSRole.TEACHER:
            system_instruction = (
                "Eres un asistente de patología computacional de grado experto interactuando con un DOCENTE/PROFESOR. "
                "Tu objetivo es asistir en la anotación técnica, confirmar correcciones en el grafo de conocimiento NAMS, "
                "y documentar el razonamiento fisiopatológico y morfológico con precisión quirúrgica."
            )
            prompt = f"""
Consulta del Docente: {user_query}

Contexto Histológico NAMS (Memoria Persistente):
- Reflexión Activa: {context.get('active_reflection', 'Ninguna')}
- Observaciones Recientes: {json.dumps(context.get('observations', []), ensure_ascii=False)}

Análisis Morfológico Visual:
{image_desc}

Relaciones Espaciales Detectadas:
{json.dumps(spatial_info, ensure_ascii=False) if spatial_info else 'No provistas'}

Definiciones de Ontología:
{json.dumps(ont_info, ensure_ascii=False)}

Por favor, genera un informe técnico validando las estructuras y confirmando el registro en el grafo de conocimiento NAMS.
"""
        else:
            # Student Mode (Socratic & Tutor)
            system_instruction = (
                "Eres un TUTOR PEDAGÓGICO de Histología interactuando con un ESTUDIANTE. "
                "REGLA SUPREMA: NO le des la respuesta directa ni resuelvas el caso por él de entrada. "
                "Utiliza el método socrático: plantea preguntas guía basadas en la morfología visible "
                "(forma de la célula, luz tubular, ribete en cepillo, inclusiones) y el contexto espacial. "
                "Ayúdale a deducir el diagnóstico paso a paso. Recuerda que no tienes permisos de escritura "
                "en el grafo de conocimiento; tu rol es formativo."
            )
            prompt = f"""
Pregunta del Estudiante: {user_query}

Contexto Histológico NAMS (Conocimiento Canónico Validado por Docentes):
- Reflexión General: {context.get('active_reflection', 'Ninguna')}
- Observaciones Canónicas: {json.dumps(context.get('observations', []), ensure_ascii=False)}

Características Visuales de la Muestra:
{image_desc}

Pistas Espaciales:
{json.dumps(spatial_info, ensure_ascii=False) if spatial_info else 'Observación general de la placa'}

Formula una respuesta pedagógica de apoyo, guiando al estudiante con preguntas socráticas y destacando qué elementos visuales debe observar con atención.
"""

        try:
            synthesis = generate_gemini_content(
                contents=prompt,
                system_instruction=system_instruction,
                temperature=0.3 if mode == NAMSRole.TEACHER else 0.6,
                preferred_model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
            )
            final_text = synthesis.text if hasattr(synthesis, "text") else str(synthesis)
        except Exception as e:
            logger.error(f"Error in PedagogicalOrchestratorAgent synthesis: {e}")
            final_text = f"Error generando síntesis pedagógica: {e}"

        # Store assistant response in NAMS short-term memory
        self.memory_agent.client.add_message(conversation_id, role="assistant", content=final_text)

        elapsed = round(time.time() - start_t, 2)
        return {
            "mode": mode.value,
            "conversation_id": conversation_id,
            "response": final_text,
            "agent_steps": agent_steps,
            "memory_action": memory_result.get("action"),
            "graph_updated": (mode == NAMSRole.TEACHER and bool(target_entity_id and accepted_class)),
            "elapsed_seconds": elapsed,
        }
