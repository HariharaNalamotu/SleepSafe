export type SessionRow = {
  session_id: string;
  start_utc: string | null;
  end_utc: string | null;
  duration_s: number | null;
  valid_audio_s: number | null;
  estimated_sleep_s: number | null;
  estimated_ahi: number | null;
  severity_estimate: string | null;
  apnea_count: number | null;
  hypopnea_count: number | null;
  snore_pct_of_sleep: number | null;
  longest_apnea_s: number | null;
  short_session: boolean;
  demo_mode: boolean;
};

export type NightEvent = {
  id?: string;
  type: string;
  start_offset_s: number;
  end_offset_s: number;
  duration_s: number;
  start_utc?: string;
  end_utc?: string;
  confidence?: number;
  confidence_basis?: string;
  reason?: string;
  evidence?: { terminated_by?: string };
};

export type NightReport = {
  session_id: string;
  model?: { name?: string; version?: string; demo_mode?: boolean };
  recording: {
    start_utc: string;
    end_utc: string;
    duration_s: number;
    valid_audio_s: number;
    estimated_sleep_s: number;
    excluded_s?: Record<string, number>;
  };
  summary: {
    estimated_ahi: number | null;
    severity_estimate: string | null;
    rate_basis?: string;
    short_session?: boolean;
    counts: Record<string, number>;
    snore_pct_of_sleep: number | null;
    longest_apnea_s: number;
    events_per_hour_by_hour?: (number | null)[];
  };
  events: NightEvent[];
  caveats: string[];
};

export type WrittenReport = { markdown: string; llm_endpoint: string | null; created_utc: string | null };
