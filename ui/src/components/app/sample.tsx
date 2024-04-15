import { cn } from "@/lib/utils";
import { mergeUint8Arrays } from "@/utils/array";

export type SimpleSampleAreaProps = {
  sample: { context: Uint8Array[] };
  sampleName: string;
  tokenGroupClassName?: (tokens: { token: Uint8Array }[], i: number) => string;
  tokenGroupProps?: (tokens: { token: Uint8Array }[], i: number) => React.HTMLProps<HTMLSpanElement>;
};

export const SimpleSampleArea = ({
  sample,
  sampleName,
  tokenGroupClassName,
  tokenGroupProps,
}: SimpleSampleAreaProps) => {
  const decoder = new TextDecoder("utf-8", { fatal: true });

  const start = Math.max(0);
  const end = Math.min(sample.context.length);
  const tokens = sample.context.slice(start, end).map((token) => ({
    token,
  }));

  type Token = { token: Uint8Array };

  const [tokenGroups, _] = tokens.reduce<[Token[][], Token[]]>(
    ([groups, currentGroup], token) => {
      const newGroup = [...currentGroup, token];
      try {
        decoder.decode(mergeUint8Arrays(newGroup.map((t) => t.token)));
        return [[...groups, newGroup], []];
      } catch {
        return [groups, newGroup];
      }
    },
    [[], []]
  );

  return (
    <div>
      {sampleName && <span className="text-gray-700 font-bold">{sampleName}: </span>}
      {tokenGroups.map((tokens, i) => (
        <span
          className={cn(
            "underline decoration-slate-400 decoration-1 decoration-dotted underline-offset-[6px]",
            tokenGroupClassName?.(tokens, i)
          )}
          key={i}
          {...tokenGroupProps?.(tokens, i)}
        >
          {decoder
            .decode(mergeUint8Arrays(tokens.map((t) => t.token)))
            .replace("\n", "⏎")
            .replace("\t", "⇥")
            .replace("\r", "↵")}
        </span>
      ))}
    </div>
  );
};
