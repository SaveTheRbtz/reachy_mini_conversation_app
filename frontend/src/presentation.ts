const avatars: Readonly<Record<string, string>> = {
  bored_teenager: "bored-teenager",
  captain_circuit: "captain-circuit",
  chess_coach: "chess-coach",
  cosmic_kitchen: "cosmic-kitchen",
  default: "default",
  hype_bot: "hype-bot",
  mad_scientist_assistant: "mad-scientist",
  mars_rover: "mars-rover",
  nature_documentarian: "nature-doc",
  noir_detective: "noir-detective",
  sorry_bro: "sorry-bro",
  time_traveler: "time-traveler",
  victorian_butler: "victorian-butler",
};

export function profileLabel(name: string): string {
  return name
    .replace(/^profiles\/(?:builtin-|user-)/, "")
    .replace(/[_-]/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function profileAvatar(name: string): string {
  return `${import.meta.env.BASE_URL}avatars/${avatars[name.replace("profiles/builtin-", "")] ?? "default"}.svg`;
}

export function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}
