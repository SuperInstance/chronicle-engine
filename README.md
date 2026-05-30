# Chronicle Engine

Universal distillation engine — assistants, people, characters, musicians, all the same math.

## What It Does

The Chronicle Engine takes any "subject" (a person, a character, a work style, a musician's style) and distills it into a tiny model through observation, simulation, ranking, and iterative refinement.

## Subject Kinds

| Kind | Purpose |
|------|---------|
| Person | Real people (interview practice, collaboration) |
| Character | Fictional characters (storytelling) |
| Assistant | Work assistants (productivity) |
| Musician | Musical styles (jam sessions) |
| MCP | MCP servers (automatic distillation) |

## Core Concepts

- **Observation** — Record what a subject actually does
- **Simulation** — Generate synthetic behavior via seeded replay
- **Ranking** — Learn user preferences from feedback
- **Distillation** — Compress observations into a compact style embedding (16 floats)

## Usage

```rust
use chronicle_engine::{ChronicleEngine, SubjectKind};

let mut engine = ChronicleEngine::new();
engine.create_subject("alice", SubjectKind::Person);
engine.observe("alice", "How are you?", "Doing great!", &[]);
engine.rank("alice", "sim_0", 0.9, Some("friendly tone"));

let distilled = engine.distill("alice").unwrap();
println!("Confidence: {}", distilled.confidence);
```

## Building

```bash
cargo build
cargo test
```

Pure Rust, zero dependencies.
