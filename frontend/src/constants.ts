export const AVATAR_BY_PROFILE: Readonly<Record<string, string>> = {
  bored_teenager: "bored-teenager.svg",
  captain_circuit: "captain-circuit.svg",
  chess_coach: "chess-coach.svg",
  cosmic_kitchen: "cosmic-kitchen.svg",
  default: "default.svg",
  hype_bot: "hype-bot.svg",
  mad_scientist_assistant: "mad-scientist.svg",
  mars_rover: "mars-rover.svg",
  nature_documentarian: "nature-doc.svg",
  noir_detective: "noir-detective.svg",
  sorry_bro: "sorry-bro.svg",
  time_traveler: "time-traveler.svg",
  victorian_butler: "victorian-butler.svg",
};

export const ORB_STATES = Object.freeze({
  MUTED: "muted",
  IDLE: "idle",
  CONNECTING: "connecting",
  ERROR: "error",
});

export type OrbState = typeof ORB_STATES[keyof typeof ORB_STATES];

export const GLOW_BY_STATE: Record<OrbState, string> = {
  [ORB_STATES.MUTED]: "#94a3b8",      // microphone muted
  [ORB_STATES.IDLE]: "#34d399",       // ready / breathing
  [ORB_STATES.CONNECTING]: "#facc15", // negotiating
  [ORB_STATES.ERROR]: "#ff6a75",
};

export const ROUTES = Object.freeze({
  TALK: "#/",
  PERSONALITIES: "#/personalities",
  SETTINGS: "#/settings",
  TOOLS: "#/tools",
});

export function avatarFor(profileName: string): string {
  const file = AVATAR_BY_PROFILE[profileName] ?? "default.svg";
  return `/static/avatars/${file}`;
}
