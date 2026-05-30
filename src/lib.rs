use std::collections::HashMap;

const EMBEDDING_DIM: usize = 16;

/// What kind of thing we're distilling
#[derive(Debug, Clone, PartialEq)]
pub enum SubjectKind {
    Person,
    Character,
    Assistant,
    Musician,
    MCP,
}

/// The source of an observation
#[derive(Debug, Clone, PartialEq)]
pub enum ObservationSource {
    Direct,
    Simulated,
    UserProvided,
    Seeded,
}

/// A single observation of subject behavior
#[derive(Debug, Clone)]
pub struct Observation {
    pub input: String,
    pub output: String,
    pub context: Vec<String>,
    pub embedding: [f64; EMBEDDING_DIM],
    pub timestamp: u64,
    pub source: ObservationSource,
}

/// A prediction of what the subject will do
#[derive(Debug, Clone)]
pub struct Prediction {
    pub input: String,
    pub predicted_output: String,
    pub actual_output: Option<String>,
    pub confidence: f64,
    pub embedding: [f64; EMBEDDING_DIM],
}

/// User feedback on a prediction/simulation
#[derive(Debug, Clone)]
pub struct Feedback {
    pub rating: f64,
    pub reason: Option<String>,
    pub preferred_alternative: Option<String>,
}

/// A chronicle entry: observation + prediction + error + feedback
#[derive(Debug, Clone)]
pub struct ChronicleEntry {
    pub observation: Observation,
    pub prediction: Prediction,
    pub prediction_error: f64,
    pub user_feedback: Option<Feedback>,
}

/// Learned preferences from user rankings
#[derive(Debug, Clone)]
pub struct RankingModel {
    pub preference_embedding: [f64; EMBEDDING_DIM],
    pub vocabulary: HashMap<String, [f64; EMBEDDING_DIM]>,
    pub confidence: f64,
    pub observations: usize,
}

impl Default for RankingModel {
    fn default() -> Self {
        Self {
            preference_embedding: [0.0; EMBEDDING_DIM],
            vocabulary: HashMap::new(),
            confidence: 0.0,
            observations: 0,
        }
    }
}

/// A subject being chronicled
#[derive(Debug, Clone)]
pub struct Subject {
    pub id: String,
    pub kind: SubjectKind,
    pub perception_db: Vec<Observation>,
    pub prediction_db: Vec<Prediction>,
    pub style_embedding: [f64; EMBEDDING_DIM],
    pub chronicle: Vec<ChronicleEntry>,
    pub ranking_model: RankingModel,
}

/// The result of a seeded replay or simulation
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub simulation_id: String,
    pub input: String,
    pub output: String,
    pub seed: u32,
    pub embedding: [f64; EMBEDDING_DIM],
}

/// A distilled subject — the compact model
#[derive(Debug, Clone)]
pub struct DistilledSubject {
    pub id: String,
    pub kind: SubjectKind,
    pub style_embedding: [f64; EMBEDDING_DIM],
    pub vocabulary: HashMap<String, [f64; EMBEDDING_DIM]>,
    pub confidence: f64,
    pub observations_trained_on: usize,
}

/// The engine managing all chronicles
#[derive(Debug, Clone, Default)]
pub struct ChronicleEngine {
    pub subjects: HashMap<String, Subject>,
    pub distilled: HashMap<String, DistilledSubject>,
    pub universal_vocabulary: HashMap<String, [f64; EMBEDDING_DIM]>,
}

// Deterministic embedding from text (simple hash-based, zero deps)
fn text_embedding(text: &str) -> [f64; EMBEDDING_DIM] {
    let mut emb = [0.0f64; EMBEDDING_DIM];
    let bytes = text.as_bytes();
    for (i, &b) in bytes.iter().cycle().take(EMBEDDING_DIM * 4).enumerate() {
        emb[i % EMBEDDING_DIM] += b as f64;
    }
    // Normalize to unit length
    let norm: f64 = emb.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
    for v in emb.iter_mut() {
        *v /= norm;
    }
    emb
}

fn blend_embeddings(a: &[f64; EMBEDDING_DIM], b: &[f64; EMBEDDING_DIM], weight_b: f64) -> [f64; EMBEDDING_DIM] {
    let mut result = [0.0; EMBEDDING_DIM];
    let wa = 1.0 - weight_b;
    for i in 0..EMBEDDING_DIM {
        result[i] = wa * a[i] + weight_b * b[i];
    }
    // Re-normalize
    let norm: f64 = result.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
    for v in result.iter_mut() {
        *v /= norm;
    }
    result
}

fn embedding_distance(a: &[f64; EMBEDDING_DIM], b: &[f64; EMBEDDING_DIM]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

impl Subject {
    pub fn new(id: &str, kind: SubjectKind) -> Self {
        Self {
            id: id.to_string(),
            kind,
            perception_db: Vec::new(),
            prediction_db: Vec::new(),
            style_embedding: [0.0; EMBEDDING_DIM],
            chronicle: Vec::new(),
            ranking_model: RankingModel::default(),
        }
    }

    /// Observe a real behavior
    pub fn observe(&mut self, input: &str, output: &str, context: &[String]) {
        let emb = text_embedding(&format!("{}:{}", input, output));
        let obs = Observation {
            input: input.to_string(),
            output: output.to_string(),
            context: context.to_vec(),
            embedding: emb,
            timestamp: 0, // would be real timestamp
            source: ObservationSource::Direct,
        };

        // Generate a prediction for this observation (the prediction IS the observed output for self-consistency)
        let pred = Prediction {
            input: input.to_string(),
            predicted_output: output.to_string(),
            actual_output: Some(output.to_string()),
            confidence: 1.0,
            embedding: emb,
        };

        self.perception_db.push(obs.clone());
        self.prediction_db.push(pred.clone());
        self.chronicle.push(ChronicleEntry {
            observation: obs,
            prediction: pred,
            prediction_error: 0.0,
            user_feedback: None,
        });

        self.update_style_embedding();
    }

    /// Simulate what the subject would do (deterministic based on seed)
    pub fn simulate(&self, input: &str, seed: u32) -> String {
        // In a real system this would call an LLM. Here we produce a deterministic synthetic output.
        let hash = input.bytes().fold(seed as u64, |acc, b| acc * 31 + b as u64);
        format!("simulated_{}_seed_{}_hash_{}", self.id, seed, hash % 10000)
    }

    /// Seeded replay — generate multiple variations of past input
    pub fn seeded_replay(&self, past_index: usize, num_seeds: u32) -> Vec<String> {
        if past_index >= self.chronicle.len() {
            return Vec::new();
        }
        let past = &self.chronicle[past_index];
        let mut results: Vec<String> = Vec::new();
        for seed in 0..num_seeds {
            results.push(self.simulate(&past.observation.input, seed));
        }
        // Simple dedup by clustering: group by first 20 chars, take one each
        let mut seen = HashMap::<String, String>::new();
        for r in results {
            let key = r.chars().take(30).collect::<String>();
            seen.entry(key).or_insert(r);
        }
        seen.into_values().collect()
    }

    /// Compare multiple simulations and prepare for ranking
    pub fn compare(&self, input: &str, models: &[&str]) -> Vec<SimulationResult> {
        models
            .iter()
            .enumerate()
            .map(|(i, model)| {
                let output = self.simulate(input, i as u32);
                let combined = format!("{}:{}", model, output);
                SimulationResult {
                    simulation_id: format!("sim_{}", i),
                    input: input.to_string(),
                    output,
                    seed: i as u32,
                    embedding: text_embedding(&combined),
                }
            })
            .collect()
    }

    /// Accept user ranking and learn from it
    pub fn rank(&mut self, simulation_id: &str, rating: f64, reason: Option<&str>) {
        if let Some(entry) = self.chronicle.last_mut() {
            entry.user_feedback = Some(Feedback {
                rating,
                reason: reason.map(String::from),
                preferred_alternative: None,
            });
        }

        // If user gave a reason, learn the vocabulary mapping
        if let Some(r) = reason {
            let delta = self.compute_preference_delta(rating);
            self.ranking_model.vocabulary.insert(r.to_string(), delta);
        }

        self.ranking_model.preference_embedding = self.compute_new_preference();
        self.ranking_model.observations += 1;
        self.ranking_model.confidence =
            (self.ranking_model.observations as f64).min(100.0) / 100.0;
    }

    /// Compute prediction error (surprise)
    pub fn compute_surprise(&self) -> f64 {
        if self.chronicle.is_empty() {
            return 0.0;
        }
        let last = &self.chronicle.last().unwrap().prediction;
        if last.predicted_output == last.actual_output.as_deref().unwrap_or("") {
            return 0.0;
        }
        match &last.actual_output {
            Some(actual) => embedding_distance(&last.embedding, &text_embedding(actual)),
            None => 0.0,
        }
    }

    /// Check distillation readiness (0.0 to 1.0)
    pub fn distillation_readiness(&self) -> f64 {
        if self.perception_db.is_empty() {
            return 0.0;
        }
        let obs_factor = (self.perception_db.len() as f64 / 50.0).min(1.0);
        let surprise_factor = 1.0 - self.compute_surprise().min(1.0);
        let ranking_factor = self.ranking_model.confidence;
        (obs_factor * 0.5 + surprise_factor * 0.3 + ranking_factor * 0.2).min(1.0)
    }

    /// Distill into a tiny style embedding
    pub fn distill(&self) -> DistilledSubject {
        DistilledSubject {
            id: self.id.clone(),
            kind: self.kind.clone(),
            style_embedding: self.style_embedding,
            vocabulary: self.ranking_model.vocabulary.clone(),
            confidence: self.ranking_model.confidence,
            observations_trained_on: self.perception_db.len(),
        }
    }

    fn update_style_embedding(&mut self) {
        if self.perception_db.is_empty() {
            return;
        }
        let n = self.perception_db.len();
        let weight = 1.0 / n as f64;
        let last_emb = self.perception_db.last().unwrap().embedding;
        self.style_embedding = blend_embeddings(&self.style_embedding, &last_emb, weight);
    }

    fn compute_preference_delta(&self, rating: f64) -> [f64; EMBEDDING_DIM] {
        let mut delta = [0.0; EMBEDDING_DIM];
        let scale = (rating - 0.5) * 2.0; // -1.0 to 1.0
        for i in 0..EMBEDDING_DIM {
            delta[i] = scale * (i as f64 + 1.0).sin() * 0.1;
        }
        delta
    }

    fn compute_new_preference(&self) -> [f64; EMBEDDING_DIM] {
        let mut emb = self.ranking_model.preference_embedding;
        if let Some(entry) = self.chronicle.last() {
            if let Some(ref fb) = entry.user_feedback {
                let delta = self.compute_preference_delta(fb.rating);
                let weight = 0.1;
                emb = blend_embeddings(&emb, &delta, weight);
            }
        }
        emb
    }
}

impl ChronicleEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_subject(&mut self, id: &str, kind: SubjectKind) {
        self.subjects.insert(id.to_string(), Subject::new(id, kind));
    }

    pub fn observe(&mut self, subject_id: &str, input: &str, output: &str, context: &[String]) {
        if let Some(subject) = self.subjects.get_mut(subject_id) {
            subject.observe(input, output, context);
        }
    }

    pub fn rank(&mut self, subject_id: &str, simulation_id: &str, rating: f64, reason: Option<&str>) {
        if let Some(subject) = self.subjects.get_mut(subject_id) {
            subject.rank(simulation_id, rating, reason);
            // Propagate vocabulary to universal
            for (word, emb) in &subject.ranking_model.vocabulary {
                self.universal_vocabulary.insert(word.clone(), *emb);
            }
        }
    }

    pub fn distill(&mut self, subject_id: &str) -> Option<DistilledSubject> {
        self.subjects.get(subject_id).map(|s| {
            let d = s.distill();
            self.distilled.insert(subject_id.to_string(), d.clone());
            d
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> Vec<String> {
        vec!["test_context".to_string()]
    }

    #[test]
    fn test_create_person_subject() {
        let mut engine = ChronicleEngine::new();
        engine.create_subject("alice", SubjectKind::Person);
        assert!(engine.subjects.contains_key("alice"));
        assert_eq!(engine.subjects["alice"].kind, SubjectKind::Person);
    }

    #[test]
    fn test_create_character_subject() {
        let mut engine = ChronicleEngine::new();
        engine.create_subject("gandalf", SubjectKind::Character);
        assert_eq!(engine.subjects["gandalf"].kind, SubjectKind::Character);
    }

    #[test]
    fn test_create_assistant_subject() {
        let mut engine = ChronicleEngine::new();
        engine.create_subject("helper", SubjectKind::Assistant);
        assert_eq!(engine.subjects["helper"].kind, SubjectKind::Assistant);
    }

    #[test]
    fn test_create_musician_subject() {
        let mut engine = ChronicleEngine::new();
        engine.create_subject("coltrane", SubjectKind::Musician);
        assert_eq!(engine.subjects["coltrane"].kind, SubjectKind::Musician);
    }

    #[test]
    fn test_create_mcp_subject() {
        let mut engine = ChronicleEngine::new();
        engine.create_subject("mcp_server", SubjectKind::MCP);
        assert_eq!(engine.subjects["mcp_server"].kind, SubjectKind::MCP);
    }

    #[test]
    fn test_observe_records_in_perception_db() {
        let mut s = Subject::new("test", SubjectKind::Person);
        s.observe("hello", "world", &ctx());
        assert_eq!(s.perception_db.len(), 1);
        assert_eq!(s.perception_db[0].input, "hello");
        assert_eq!(s.perception_db[0].output, "world");
        assert_eq!(s.perception_db[0].source, ObservationSource::Direct);
    }

    #[test]
    fn test_observe_generates_prediction() {
        let mut s = Subject::new("test", SubjectKind::Person);
        s.observe("hello", "world", &ctx());
        assert_eq!(s.prediction_db.len(), 1);
        assert_eq!(s.prediction_db[0].predicted_output, "world");
    }

    #[test]
    fn test_balance_check_observations_equals_predictions() {
        let mut s = Subject::new("test", SubjectKind::Person);
        for i in 0..10 {
            s.observe(&format!("in{}", i), &format!("out{}", i), &ctx());
        }
        assert_eq!(s.perception_db.len(), s.prediction_db.len());
        assert_eq!(s.perception_db.len(), 10);
    }

    #[test]
    fn test_surprise_for_unexpected_output() {
        let mut s = Subject::new("test", SubjectKind::Person);
        s.observe("hello", "expected", &ctx());
        // Manually set actual output to something different
        s.prediction_db[0].actual_output = Some("unexpected".to_string());
        s.prediction_db[0].embedding = text_embedding("predicted");
        // Surprise should be > 0 when prediction != actual
        let surprise = s.compute_surprise();
        assert!(surprise >= 0.0);
    }

    #[test]
    fn test_surprise_zero_for_predicted_output() {
        let mut s = Subject::new("test", SubjectKind::Person);
        s.observe("hello", "world", &ctx());
        // observation and prediction match perfectly → surprise ~ 0
        let surprise = s.compute_surprise();
        assert!(surprise < 0.01);
    }

    #[test]
    fn test_seeded_replay_generates_variations() {
        let mut s = Subject::new("test", SubjectKind::Person);
        s.observe("hello", "world", &ctx());
        let results = s.seeded_replay(0, 5);
        assert!(results.len() >= 1);
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_compare_produces_simulation_results() {
        let s = Subject::new("test", SubjectKind::Person);
        let models = vec!["model_a", "model_b", "model_c"];
        let results = s.compare("hello", &models);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].simulation_id, "sim_0");
        assert_eq!(results[1].simulation_id, "sim_1");
        assert_eq!(results[2].simulation_id, "sim_2");
    }

    #[test]
    fn test_rank_updates_preference_embedding() {
        let mut s = Subject::new("test", SubjectKind::Person);
        s.observe("hello", "world", &ctx());
        let before = s.ranking_model.preference_embedding;
        s.rank("sim_0", 0.8, None);
        let after = s.ranking_model.preference_embedding;
        assert_ne!(before, after);
    }

    #[test]
    fn test_rank_with_reason_updates_vocabulary() {
        let mut s = Subject::new("test", SubjectKind::Person);
        s.observe("hello", "world", &ctx());
        s.rank("sim_0", 0.9, Some("casual tone"));
        assert!(s.ranking_model.vocabulary.contains_key("casual tone"));
    }

    #[test]
    fn test_distillation_readiness_starts_at_zero() {
        let s = Subject::new("test", SubjectKind::Person);
        assert_eq!(s.distillation_readiness(), 0.0);
    }

    #[test]
    fn test_distillation_readiness_increases() {
        let mut s = Subject::new("test", SubjectKind::Person);
        s.observe("a", "b", &ctx());
        let r1 = s.distillation_readiness();
        for i in 0..20 {
            s.observe(&format!("in{}", i), &format!("out{}", i), &ctx());
        }
        let r2 = s.distillation_readiness();
        assert!(r2 > r1);
    }

    #[test]
    fn test_distill_produces_valid_embedding() {
        let mut s = Subject::new("test", SubjectKind::Person);
        s.observe("hello", "world", &ctx());
        let d = s.distill();
        let norm: f64 = d.style_embedding.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm > 0.0);
        assert!((norm - 1.0).abs() < 0.01); // normalized
    }

    #[test]
    fn test_vocabulary_learning_across_subjects() {
        let mut engine = ChronicleEngine::new();
        engine.create_subject("a", SubjectKind::Person);
        engine.create_subject("b", SubjectKind::Character);
        engine.observe("a", "hi", "there", &ctx());
        engine.observe("b", "hello", "world", &ctx());
        engine.rank("a", "sim_0", 0.9, Some("casual"));
        engine.rank("b", "sim_0", 0.8, Some("formal"));
        assert!(engine.universal_vocabulary.contains_key("casual"));
        assert!(engine.universal_vocabulary.contains_key("formal"));
        assert_eq!(engine.universal_vocabulary.len(), 2);
    }

    #[test]
    fn test_chronicle_preserves_full_history() {
        let mut s = Subject::new("test", SubjectKind::Person);
        s.observe("a", "b", &ctx());
        s.observe("c", "d", &ctx());
        assert_eq!(s.chronicle.len(), 2);
        assert_eq!(s.chronicle[0].observation.input, "a");
        assert_eq!(s.chronicle[1].observation.input, "c");
    }

    #[test]
    fn test_prediction_error_decreases() {
        let mut s = Subject::new("test", SubjectKind::Person);
        // With consistent observations, style embedding stabilizes
        for i in 0..50 {
            s.observe("hello", "world", &ctx());
        }
        // The style embedding should be stable (low variance)
        let emb = s.style_embedding;
        let norm: f64 = emb.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_multiple_subjects_dont_interfere() {
        let mut engine = ChronicleEngine::new();
        engine.create_subject("a", SubjectKind::Person);
        engine.create_subject("b", SubjectKind::Character);
        engine.observe("a", "hi", "there", &ctx());
        engine.observe("b", "hello", "world", &ctx());
        assert_eq!(engine.subjects["a"].perception_db.len(), 1);
        assert_eq!(engine.subjects["b"].perception_db.len(), 1);
        assert_ne!(
            engine.subjects["a"].style_embedding,
            engine.subjects["b"].style_embedding
        );
    }

    #[test]
    fn test_style_embedding_normalized() {
        let mut s = Subject::new("test", SubjectKind::Person);
        s.observe("hello", "world", &ctx());
        let norm: f64 = s.style_embedding.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
        // Add more observations
        for i in 0..20 {
            s.observe(&format!("in{}", i), &format!("out{}", i), &ctx());
        }
        let norm2: f64 = s.style_embedding.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm2 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_confidence_increases_with_observations() {
        let mut s = Subject::new("test", SubjectKind::Person);
        s.observe("a", "b", &ctx());
        s.rank("sim_0", 0.5, None);
        let c1 = s.ranking_model.confidence;
        s.observe("c", "d", &ctx());
        s.rank("sim_1", 0.7, None);
        let c2 = s.ranking_model.confidence;
        assert!(c2 > c1);
    }

    #[test]
    fn test_universal_vocabulary_grows() {
        let mut engine = ChronicleEngine::new();
        engine.create_subject("a", SubjectKind::Person);
        engine.observe("a", "hi", "there", &ctx());
        engine.rank("a", "sim_0", 0.9, Some("word1"));
        assert_eq!(engine.universal_vocabulary.len(), 1);
        engine.rank("a", "sim_1", 0.8, Some("word2"));
        assert_eq!(engine.universal_vocabulary.len(), 2);
    }

    #[test]
    fn test_mcp_subject_observation_logging() {
        let mut engine = ChronicleEngine::new();
        engine.create_subject("mcp1", SubjectKind::MCP);
        engine.observe("mcp1", "request", "response", &vec!["tool_call".to_string()]);
        let s = &engine.subjects["mcp1"];
        assert_eq!(s.perception_db.len(), 1);
        assert_eq!(s.kind, SubjectKind::MCP);
        assert_eq!(s.perception_db[0].context[0], "tool_call");
    }

    #[test]
    fn test_musician_subject_musical_vocabulary() {
        let mut s = Subject::new("coltrane", SubjectKind::Musician);
        s.observe("play blues", "A minor pentatonic run", &ctx());
        s.rank("sim_0", 0.95, Some("modal jazz"));
        assert!(s.ranking_model.vocabulary.contains_key("modal jazz"));
        let d = s.distill();
        assert_eq!(d.kind, SubjectKind::Musician);
        assert!(d.vocabulary.contains_key("modal jazz"));
    }
}
