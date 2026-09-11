export interface ConversationStatus {
  model: string;
  has_key: boolean;
  connected: boolean;
  connection_state: "not_started" | "waiting_for_config" | "connecting" | "connected" | "disconnected";
  connection_error: string | null;
  voice: string;
  muted: boolean;
  playing: boolean;
}

export interface PersonalityList {
  choices: string[];
  current: string;
  startup: string;
  locked: boolean;
  locked_to: string | null;
}

export interface Personality {
  instructions: string;
  greeting: string;
  tools_text: string;
  voice: string;
  uses_default_voice: boolean;
  available_tools: string[];
  enabled_tools: string[];
}

export interface PersonalitySave {
  name: string;
  instructions: string;
  voice?: string | null;
  greeting?: string | null;
  tools_text?: string;
  overwrite?: boolean;
}

export interface AvailableTool {
  id: string;
  description: string;
}

export interface ProfileTools {
  profile: string;
  is_active: boolean;
  overridden: boolean;
  editable: boolean;
  profiles: { id: string; active: boolean }[];
  enabled_tools: string[];
  available_tools: AvailableTool[];
  unavailable_enabled_tools: string[];
}

export interface RpcMethods {
  "conversation.status": { params: Record<string, never>; result: ConversationStatus };
  "conversation.say": { params: { text: string }; result: { ok: true } };
  "conversation.interrupt": { params: Record<string, never>; result: { ok: true } };
  "conversation.mic": { params: { muted?: boolean }; result: { muted: boolean } };
  "openai.config": { params: { api_key: string }; result: ConversationStatus };
  "personalities.list": { params: Record<string, never>; result: PersonalityList };
  "personalities.load": { params: { name: string }; result: Personality };
  "personalities.save": { params: PersonalitySave; result: { ok: true; value: string; choices: string[] } };
  "personalities.delete": { params: { name: string }; result: { ok: true; choices: string[] } };
  "personalities.apply": {
    params: { name: string; persist?: boolean; force?: boolean };
    result: { ok: true; status: string; startup: string };
  };
  "voices.list": { params: Record<string, never>; result: string[] };
  "voices.current": { params: Record<string, never>; result: { voice: string } };
  "voices.apply": { params: { voice: string }; result: { ok: true; status: string } };
  "profile_tools.get": { params: { profile?: string | null }; result: ProfileTools };
  "profile_tools.save": {
    params: { profile: string; enabled_tools: string[] };
    result: ProfileTools & { message: string };
  };
  "profile_tools.reset": { params: { profile: string }; result: ProfileTools & { message: string } };
}

export interface RpcNotifications {
  "rpc.connection": { connected: boolean };
  "conversation.activity": { reason: "connected" | "disconnected" | "playback_started" | "playback_stopped" };
}
