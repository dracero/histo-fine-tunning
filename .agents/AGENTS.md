# AI Code Review Guidelines

These rules apply to any code reviews or code modification tasks performed by AI agents in this workspace.

## 1. General Review Principles
- **Conciseness and Clarity**: Code reviews must be direct, outlining the problem, the rationale, and the concrete solution.
- **Actionable Feedback**: Always provide code suggestions using git-style diffs or standard code blocks when presenting modifications.
- **Safety First**: Verify that no API keys, secrets, or passwords are hardcoded or committed to git.

## 2. Python & Agent2Agent (A2A) SDK
- **Type Safety**: Ensure type annotations are used for new functions and parameters.
- **A2A Protocol**: Ensure proper usage of the `a2a-sdk` (matching version `>=0.3.0,<1.0.0`).
  - Check that agent endpoints process payloads correctly and return the expected Server-Sent Events (SSE) or JSON responses.
  - Verify that authentications/JWT tokens are correctly validated where necessary.
- **Error Handling**: Use explicit try-except blocks. Avoid catching generic `Exception` unless logging it properly or re-raising it.

## 3. GPU / PyTorch Performance (RTX 3060 / CUDA)
Since this repository runs workloads optimized for GPU devices (like GPD 3060 laptop GPU):
- **Memory Management**:
  - Avoid memory leaks: use `torch.no_grad()` or `with torch.inference_mode():` for inference.
  - Use `torch.cuda.empty_cache()` where appropriate if memory limits are close.
- **Data Transfer**:
  - Minimize CPU <-> GPU transfers.
  - Move tensors to the GPU (`.to(device)`) in batch instead of one-by-one.
  - Keep models and cache weights resident in VRAM.

## 4. Frontend & Next.js / TypeScript
- **TypeScript Strictness**: Avoid the use of `any` types. Provide interfaces/types for all properties and states.
- **Responsive & Premium Design**:
  - CSS/Tailwind classes should follow standard responsive utilities.
  - Maintain a clean aesthetic matching the current premium visual system.
- **Performance**: Use React state hooks and Next.js APIs optimally to minimize re-renders.

## 5. Complexity & Database Optimization
- **Time & Space Complexity**:
  - Review algorithmic time and space complexity.
  - Optimize operations to aim for $O(\log n)$ (or similar optimal complexity) where search, indexing, or lookups are involved.
  - Avoid nested loops that lead to $O(n^2)$ or worse when processing large datasets.
- **Database Optimization**:
  - Verify that database queries (e.g., SQLite, Neo4j, Qdrant) are optimized.
  - Ensure correct indexing on queried columns/fields.
  - Avoid N+1 query problems by using batch queries, eager loading, or aggregation where appropriate.
