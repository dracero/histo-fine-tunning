#!/usr/bin/env python3
"""
Gemini + NAMS Interactive CLI for Histological Multi-Agent Analysis.

Usage examples:
  # 1. Modo Profesor (anotación, validación y escritura en grafo NAMS):
  python scripts/gemini_nams_cli.py --mode teacher --query "Identifica las células del glomérulo central"

  # 2. Modo Profesor aplicando una corrección que se guarda en el grafo de NAMS:
  python scripts/gemini_nams_cli.py --mode teacher --target cell_4 --correct "Podocito" --reason "Célula estrellada con pedicelos en la capa visceral"

  # 3. Modo Estudiante (modo tutor socrático, no altera el grafo):
  python scripts/gemini_nams_cli.py --mode student --query "¿Por qué la célula 4 no es una célula endotelial?"

  # 4. Modo Interactivo (Chat sesión continua):
  python scripts/gemini_nams_cli.py --interactive --mode student

  # 5. Visualizar la memoria NAMS acumulada:
  python scripts/gemini_nams_cli.py --show-memory
"""

import argparse
import os
import sys
import json
from pathlib import Path
from PIL import Image

# Ensure root and backend are in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from backend.nams_memory import NAMSClient, NAMSRole
from backend.adk_histology_agents import PedagogicalOrchestratorAgent

# Terminal colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner(mode: str) -> None:
    print(f"{CYAN}{BOLD}==================================================================={RESET}")
    print(f"{CYAN}{BOLD}   Gemini + Neo4j Agent Memory Service (NAMS) - Histology CLI     {RESET}")
    print(f"{CYAN}{BOLD}==================================================================={RESET}")
    if mode == "teacher":
        print(f" {GREEN}{BOLD}[ROL: PROFESOR / DOCENTE]{RESET}")
        print(f" {GREEN}✔ Permisos: memory:write, entities:write, reasoning:write{RESET}")
        print(f" {GREEN}✔ Acepta correcciones humanas (HITL) y actualiza el Grafo de Conocimiento.{RESET}")
    else:
        print(f" {MAGENTA}{BOLD}[ROL: ESTUDIANTE / APRENDIZ]{RESET}")
        print(f" {MAGENTA}⚠ Permisos: memory:read, context:read (SOLO LECTURA){RESET}")
        print(f" {MAGENTA}ℹ Tutoría Socrática: Ayuda al estudiante sin alterar el Grafo de Referencia.{RESET}")
    print(f"{CYAN}-------------------------------------------------------------------{RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini + NAMS Histology Agent CLI")
    parser.add_argument("--mode", choices=["teacher", "student"], default="teacher", help="Rol pedagógico")
    parser.add_argument("--image", type=str, default=None, help="Ruta a imagen histológica")
    parser.add_argument("--query", type=str, default=None, help="Pregunta o instrucción")
    parser.add_argument("--ontology", type=str, default="arch4", help="Nombre de ontología")
    parser.add_argument("--conv-id", type=str, default="cli_histology_session", help="ID de conversación NAMS")
    parser.add_argument("--target", type=str, default=None, help="ID de estructura/célula a corregir (Modo Docente)")
    parser.add_argument("--correct", type=str, default=None, help="Clase corregida por el docente")
    parser.add_argument("--reason", type=str, default=None, help="Criterio fisiopatológico/morfológico")
    parser.add_argument("--interactive", action="store_true", help="Iniciar sesión interactiva de diálogo")
    parser.add_argument("--show-memory", action="store_true", help="Mostrar estado de la memoria NAMS")

    args = parser.parse_args()

    nams_client = NAMSClient()
    orchestrator = PedagogicalOrchestratorAgent(nams_client=nams_client)

    if args.show_memory:
        print(f"\n{YELLOW}{BOLD}=== Estado Actual de Memoria NAMS ==={RESET}")
        context = nams_client.get_context(args.conv_id)
        print(f"\n{BOLD}Reflexión Activa:{RESET}\n{context.get('active_reflection') or '(vacía)'}")
        print(f"\n{BOLD}Observaciones Recientes ({len(context.get('observations', []))}):{RESET}")
        for obs in context.get("observations", []):
            print(f" - {obs}")
        graph_vis = nams_client.get_graph_visualization()
        print(f"\n{BOLD}Grafo de Conocimiento ({graph_vis['total_nodes']} nodos, {graph_vis['total_edges']} aristas):{RESET}")
        for node in graph_vis["nodes"][:10]:
            print(f" • [{node['type']}] {node['label']}: {node['description']}")
        return

    role = NAMSRole.TEACHER if args.mode == "teacher" else NAMSRole.STUDENT
    print_banner(role.value)

    pil_img = None
    if args.image and os.path.exists(args.image):
        try:
            pil_img = Image.open(args.image).convert("RGB")
            print(f"{GREEN}✔ Imagen cargada: {args.image} ({pil_img.size[0]}x{pil_img.size[1]} px){RESET}")
        except Exception as e:
            print(f"{RED}Error cargando imagen: {e}{RESET}")

    if args.interactive:
        print(f"{YELLOW}Iniciando diálogo interactivo en modo {role.value.upper()}. Escribe 'salir' para terminar.{RESET}\n")
        while True:
            try:
                user_input = input(f"{BOLD}[Tú ({role.value})]:{RESET} ")
                if not user_input.strip():
                    continue
                if user_input.strip().lower() in ["salir", "exit", "quit"]:
                    break

                target_id = None
                corr_class = None
                corr_reason = None

                # Detect if teacher wants to correct in-line: "corregir: cell_1 -> Podocito : criterio"
                if role == NAMSRole.TEACHER and "->" in user_input and ":" in user_input:
                    try:
                        parts = user_input.split("->")
                        target_id = parts[0].replace("corregir", "").replace("corrige", "").strip()
                        subparts = parts[1].split(":")
                        corr_class = subparts[0].strip()
                        corr_reason = subparts[1].strip() if len(subparts) > 1 else "Corrección directa de docente"
                    except Exception:
                        pass

                result = orchestrator.run_cycle(
                    image=pil_img,
                    user_query=user_input,
                    mode=role,
                    conversation_id=args.conv_id,
                    ontology_name=args.ontology,
                    target_entity_id=target_id,
                    accepted_class=corr_class,
                    teacher_rationale=corr_reason,
                )

                print(f"\n{CYAN}{BOLD}--- Traza de Agentes Especialistas ---{RESET}")
                for step in result.get("agent_steps", []):
                    print(f" {YELLOW}↳ {step}{RESET}")

                if result.get("graph_updated"):
                    print(f" {GREEN}{BOLD}✔ [NAMS Knowledge Graph] Entidad {target_id} guardada en el Grafo!{RESET}")

                print(f"\n{BOLD}[Asistente Gemini NAMS]:{RESET}\n{result['response']}\n")
                print(f"{CYAN}-------------------------------------------------------------------{RESET}\n")

            except (KeyboardInterrupt, EOFError):
                print(f"\n{YELLOW}Sesión finalizada.{RESET}")
                break
        return

    # Single-shot execution
    query = args.query or ("Analiza las estructuras visibles" if not args.correct else f"Validación de {args.target}")
    result = orchestrator.run_cycle(
        image=pil_img,
        user_query=query,
        mode=role,
        conversation_id=args.conv_id,
        ontology_name=args.ontology,
        target_entity_id=args.target,
        accepted_class=args.correct,
        teacher_rationale=args.reason,
    )

    print(f"\n{CYAN}{BOLD}--- Pasos Ejecutados por Agentes ADK ---{RESET}")
    for step in result.get("agent_steps", []):
        print(f" {YELLOW}↳ {step}{RESET}")

    if result.get("graph_updated"):
        print(f" {GREEN}{BOLD}✔ [NAMS Memory] Corrección docente comiteada al Grafo de Conocimiento Neo4j!{RESET}")

    print(f"\n{BOLD}[Respuesta Final ({role.value})]:{RESET}\n{result['response']}")


if __name__ == "__main__":
    main()
