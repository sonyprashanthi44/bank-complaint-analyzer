export type ClassificationResult = {
  product: string;
  confidence: number;
  top_probabilities: Array<Record<string, number>>;
};

export type SimilarCase = {
  case_id: string;
  product: string;
  issue: string;
  narrative: string;
  resolution_summary: string;
  company_response: string;
  similarity: number;
  source: string;
};

export type DraftReply = {
  subject: string;
  body: string;
  tone: string;
  grounded_case_ids: string[];
  provider: string;
  is_fallback: boolean;
};

export type AnalyzeResponse = {
  classification: ClassificationResult;
  similar_cases: SimilarCase[];
  draft_reply: DraftReply;
  retrieval_backend: string;
  notes: string[];
};

export type HealthResponse = {
  status: string;
  app: string;
  version: string;
  classifier_loaded: boolean;
  retrieval_backend: string;
  llm_ready: boolean;
  case_count: number;
};
