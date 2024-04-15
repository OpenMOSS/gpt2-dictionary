import { cn } from "@/lib/utils";
import { DictionarySample, DictionaryToken } from "@/types/dictionary";
import { mergeUint8Arrays, zip } from "@/utils/array";
import { useState } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "../ui/data-table";
import { getAccentClassname } from "@/utils/style";
import { SimpleSampleArea } from "../app/sample";
import { HoverCard, HoverCardContent } from "../ui/hover-card";
import { HoverCardTrigger } from "@radix-ui/react-hover-card";
import { FeatureLinkWithPreview } from "../app/feature-preview";
import { Trash2 } from "lucide-react";

export type DictionarySampleAreaProps = {
  samples: DictionarySample[];
  onSamplesChange?: (samples: DictionarySample[]) => void;
  dictionaryName: string;
};

export const DictionarySampleArea = ({ samples, onSamplesChange, dictionaryName }: DictionarySampleAreaProps) => {
  const [selectedTokenGroupIndices, setSelectedTokenGroupIndices] = useState<[number, number][]>([]);
  const toggleSelectedTokenGroupIndex = (sampleIndex: number, tokenGroupIndex: number) => {
    setSelectedTokenGroupIndices((prev) =>
      prev.some(([s, t]) => s === sampleIndex && t === tokenGroupIndex)
        ? prev.filter(([s, t]) => s !== sampleIndex || t !== tokenGroupIndex)
        : ([...prev, [sampleIndex, tokenGroupIndex]] as [number, number][]).sort(
            ([s1, t1], [s2, t2]) => s1 - s2 || t1 - t2
          )
    );
  };

  const decoder = new TextDecoder("utf-8", { fatal: true });

  const tokens = samples.map((sample) =>
    sample.context.map((token, i) => ({
      token,
      featureActs: zip(sample.featureActsIndices[i], sample.featureActs[i], sample.maxFeatureActs[i]).map(
        ([featureActIndex, featureAct, maxFeatureAct]) => ({
          featureActIndex,
          featureAct,
          maxFeatureAct,
        })
      ),
    }))
  );

  const tokenGroups = tokens
    .map((t) =>
      t.reduce<[DictionaryToken[][], DictionaryToken[]]>(
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
      )
    )
    .map((v) => v[0]);

  const selectedTokenGroups = selectedTokenGroupIndices.map(([s, t]) => tokenGroups[s][t]);
  const selectedTokens = selectedTokenGroups.flatMap((tokens) => tokens);
  const columns: ColumnDef<{ featureIndex: number; [key: `token${number}`]: string }, string>[] = [
    {
      accessorKey: "featureIndex",
      header: () => (
        <div>
          <span className="font-bold">Feature</span>
          <span> \ </span>
          <span className="font-bold">Token</span>
        </div>
      ),
      cell: ({ row }) => (
        <div>
          <FeatureLinkWithPreview dictionaryName={dictionaryName} featureIndex={row.getValue("featureIndex")} />
        </div>
      ),
    },
    ...(selectedTokenGroupIndices
      .map(([s, t]) => [tokenGroups[s][t], [s, t]] as const)
      .flatMap(([tokens, i]) => tokens.map((token) => [token, i] as const))
      .map(([token, [s, t]], i) => ({
        accessorKey: `token${i}`,
        header: () => (
          <HoverCard>
            <HoverCardTrigger className="select-none cursor-pointer">
              {token.token.reduce(
                (acc, b) =>
                  b < 32 || b > 126 ? `${acc}\\x${b.toString(16).padStart(2, "0")}` : `${acc}${String.fromCharCode(b)}`,
                ""
              )}
            </HoverCardTrigger>
            <HoverCardContent className="w-auto max-w-[800px]">
              <SimpleSampleArea
                sample={samples[s]}
                sampleName={`Sample ${s + 1}`}
                tokenGroupClassName={(_, j) => (j === t ? "bg-orange-500" : "")}
              />
            </HoverCardContent>
          </HoverCard>
        ),
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        cell: ({ row }: { row: any }) => {
          const featureAct = row.getValue(`token${i}`);
          return (
            <div className={cn(getAccentClassname(featureAct.featureAct, featureAct.maxFeatureAct, "text"))}>
              {featureAct.featureAct.toFixed(3)}
            </div>
          );
        },
      })) || []),
  ];

  const data = Object.entries(
    selectedTokens
      ?.flatMap((token, i) =>
        token.featureActs.map((featureAct) => ({
          token: token.token,
          tokenIndex: i,
          ...featureAct,
        }))
      )
      .reduce((acc, featureAct) => {
        // Group by featureActIndex
        const key = featureAct.featureActIndex.toString();
        if (acc[key]) {
          acc[key].push(featureAct);
        } else {
          acc[key] = [featureAct];
        }
        return acc;
      }, {} as Record<string, { token: Uint8Array; tokenIndex: number; featureAct: number; maxFeatureAct: number }[]>) ||
      {}
  )
    .sort(
      // Sort by sum of featureAct
      ([_a, a], [_b, b]) =>
        b.reduce((acc, { featureAct }) => acc + featureAct, 0) - a.reduce((acc, { featureAct }) => acc + featureAct, 0)
    )
    .map(([featureIndex, featureActs]) => {
      return {
        featureIndex: parseInt(featureIndex),
        ...selectedTokens?.reduce(
          (acc, _, i) => ({
            ...acc,
            [`token${i}`]: featureActs.find((featureAct) => featureAct.tokenIndex === i) || {
              featureAct: 0,
              maxFeatureAct: 0,
            },
          }),
          {}
        ),
      };
    });

  return (
    <div className="flex flex-col gap-4">
      {samples.map((sample, sampleIndex) => (
        <div className="w-full overflow-x-visible relative" key={sampleIndex}>
          <SimpleSampleArea
            sample={sample}
            sampleName={`Sample ${sampleIndex + 1}`}
            tokenGroupClassName={(_, tokenIndex) =>
              cn(
                "hover:shadow-lg hover:text-gray-600 cursor-pointer",
                selectedTokenGroupIndices.some(([s, t]) => s === sampleIndex && t === tokenIndex) && "bg-orange-500"
              )
            }
            tokenGroupProps={(_, i) => ({
              onClick: () => toggleSelectedTokenGroupIndex(sampleIndex, i),
            })}
          />
          <div className="absolute top-0 -left-8 h-full opacity-20 hover:opacity-100 transition-opacity">
            <Trash2
              className="cursor-pointer hover:text-red-500 shrink-0 m-0.5"
              size={20}
              onClick={() => {
                setSelectedTokenGroupIndices((prev) =>
                  prev.filter(([s, _]) => s !== sampleIndex).map(([s, t]) => (s > sampleIndex ? [s - 1, t] : [s, t]))
                );
                onSamplesChange?.(samples.filter((_, i) => i !== sampleIndex));
              }}
            />
          </div>
        </div>
      ))}

      {selectedTokens.length > 0 && <DataTable columns={columns} data={data} />}
    </div>
  );
};
