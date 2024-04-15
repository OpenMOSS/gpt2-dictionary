import { cn } from "@/lib/utils";
import { Token } from "@/types/feature";
import { mergeUint8Arrays } from "@/utils/array";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../ui/hover-card";
import { Separator } from "../ui/separator";
import { Fragment } from "react/jsx-runtime";
import { getAccentClassname } from "@/utils/style";

export type TokenInfoProps = {
  token: Token;
  maxFeatureAct: number;
};

export const TokenInfo = ({ token, maxFeatureAct }: TokenInfoProps) => {
  const hex = token.token.reduce(
    (acc, b) => (b < 32 || b > 126 ? `${acc}\\x${b.toString(16).padStart(2, "0")}` : `${acc}${String.fromCharCode(b)}`),
    ""
  );

  return (
    <div className="grid grid-cols-2 gap-2">
      <div className="text-sm font-bold">Token:</div>
      <div className="text-sm underline whitespace-pre-wrap">{hex}</div>
      <div className="text-sm font-bold">Activation:</div>
      <div className={cn("text-sm", getAccentClassname(token.featureAct, maxFeatureAct, "text"))}>
        {token.featureAct.toFixed(3)}
      </div>
    </div>
  );
};

export type SuperTokenProps = {
  tokens: Token[];
  maxFeatureAct: number;
  sampleMaxFeatureAct: number;
};

export const SuperToken = ({ tokens, maxFeatureAct, sampleMaxFeatureAct }: SuperTokenProps) => {
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const displayText = decoder
    .decode(mergeUint8Arrays(tokens.map((t) => t.token)))
    .replace("\n", "⏎")
    .replace("\t", "⇥")
    .replace("\r", "↵");

  const superTokenMaxFeatureAct = Math.max(...tokens.map((t) => t.featureAct));

  const SuperTokenInner = () => {
    return (
      <span
        className={cn(
          "underline decoration-slate-400 decoration-1 decoration-dotted underline-offset-[6px]",
          superTokenMaxFeatureAct > 0 && "hover:shadow-lg hover:text-gray-600 cursor-pointer",
          sampleMaxFeatureAct > 0 && superTokenMaxFeatureAct == sampleMaxFeatureAct && "font-bold",
          getAccentClassname(superTokenMaxFeatureAct, maxFeatureAct, "bg")
        )}
      >
        {displayText}
      </span>
    );
  };

  if (superTokenMaxFeatureAct === 0) {
    return <SuperTokenInner />;
  }

  return (
    <HoverCard>
      <HoverCardTrigger>
        <SuperTokenInner />
      </HoverCardTrigger>
      <HoverCardContent className="w-[300px] text-wrap flex flex-col gap-4">
        {tokens.length > 1 && (
          <div className="text-sm font-bold">This super token is composed of the {tokens.length} tokens below:</div>
        )}
        {tokens.map((token, i) => (
          <Fragment key={i}>
            <TokenInfo token={token} maxFeatureAct={maxFeatureAct} />
            {i < tokens.length - 1 && <Separator />}
          </Fragment>
        ))}
      </HoverCardContent>
    </HoverCard>
  );
};
