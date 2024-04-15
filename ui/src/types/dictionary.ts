import { z } from "zod";

export const DictionarySchema = z.object({
  dictionaryName: z.string(),
  featureActivationTimesHistogram: z.any(),
  aliveFeatureCount: z.number(),
});

export type Dictionary = z.infer<typeof DictionarySchema>;

export const DictionarySampleSchema = z.object({
  context: z.array(z.instanceof(Uint8Array)),
  featureActsIndices: z.array(z.array(z.number())),
  featureActs: z.array(z.array(z.number())),
  maxFeatureActs: z.array(z.array(z.number())),
});

export type DictionarySample = z.infer<typeof DictionarySampleSchema>;

export type DictionaryToken = {
  token: Uint8Array;
  featureActs: {
    featureActIndex: number;
    featureAct: number;
    maxFeatureAct: number;
  }[];
};
