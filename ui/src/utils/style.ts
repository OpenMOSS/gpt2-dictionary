export const getAccentClassname = (featureAct: number, maxFeatureAct: number, variant: "text" | "bg") => {
  const accentClassnames = [
    null,
    variant === "text" ? "text-orange-100" : "bg-orange-100",
    variant === "text" ? "text-orange-200" : "bg-orange-200",
    variant === "text" ? "text-orange-300" : "bg-orange-300",
    variant === "text" ? "text-orange-400" : "bg-orange-400",
    variant === "text" ? "text-orange-500" : "bg-orange-500",
  ];

  return accentClassnames[Math.ceil((featureAct / maxFeatureAct) * (accentClassnames.length - 1))];
};
