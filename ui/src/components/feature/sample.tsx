import { Feature, Sample, Token } from "@/types/feature";
import { SuperToken } from "./token";
import { mergeUint8Arrays } from "@/utils/array";
import { useState } from "react";
import { AppPagination } from "../ui/pagination";

export const FeatureSampleGroup = ({
  feature,
  sampleGroup,
}: {
  feature: Feature;
  sampleGroup: Feature["sampleGroups"][0];
}) => {
  const [page, setPage] = useState<number>(1);
  const maxPage = Math.ceil(sampleGroup.samples.length / 5);

  return (
    <div className="flex flex-col gap-4 mt-4">
      <p className="font-bold">Max Activation: {Math.max(...sampleGroup.samples[0].featureActs).toFixed(3)}</p>
      {sampleGroup.samples.slice((page - 1) * 5, page * 5).map((sample, i) => (
        <FeatureActivationSample
          key={i}
          sample={sample}
          sampleName={`Sample ${(page - 1) * 5 + i + 1}`}
          maxFeatureAct={feature.maxFeatureAct}
        />
      ))}
      <AppPagination page={page} setPage={setPage} maxPage={maxPage} />
    </div>
  );
};

export type FeatureActivationSampleProps = {
  sample: Sample;
  sampleName: string;
  maxFeatureAct: number;
};

export const FeatureActivationSample = ({ sample, sampleName, maxFeatureAct }: FeatureActivationSampleProps) => {
  const sampleMaxFeatureAct = Math.max(...sample.featureActs);

  const decoder = new TextDecoder("utf-8", { fatal: true });

  const start = Math.max(0);
  const end = Math.min(sample.context.length);
  const tokens = sample.context.slice(start, end).map((token, i) => ({
    token,
    featureAct: sample.featureActs[start + i],
  }));

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
        <SuperToken
          key={`group-${i}`}
          tokens={tokens}
          maxFeatureAct={maxFeatureAct}
          sampleMaxFeatureAct={sampleMaxFeatureAct}
        />
      ))}
    </div>
  );
};
