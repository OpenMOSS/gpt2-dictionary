import { z } from "zod";

export const SampleSchema = z.object({
  context: z.array(z.instanceof(Uint8Array)),
  featureActs: z.array(z.number()),
});

export type Sample = z.infer<typeof SampleSchema>;

export const InterpretationSchema = z.object({
  text: z.string(),
  validation: z.array(
    z.object({
      method: z.string(),
      passed: z.boolean(),
      detail: z
        .object({
          prompt: z.string(),
          response: z.string(),
        })
        .optional(),
    })
  ),
  detail: z
    .object({
      prompt: z.string(),
      response: z.string(),
    })
    .optional(),
});

export type Interpretation = z.infer<typeof InterpretationSchema>;

export const FeatureSchema = z.object({
  featureIndex: z.number(),
  dictionaryName: z.string(),
  featureActivationHistogram: z.any(),
  actTimes: z.number(),
  maxFeatureAct: z.number(),
  sampleGroups: z.array(
    z.object({
      analysisName: z.string(),
      samples: z.array(SampleSchema),
    })
  ),
  logits: z
    .object({
      topPositive: z.array(
        z.object({
          logit: z.number(),
          token: z.string(),
        })
      ),
      topNegative: z.array(
        z.object({
          logit: z.number(),
          token: z.string(),
        })
      ),
      histogram: z.any(),
    })
    .nullable(),
  interpretation: InterpretationSchema.nullable(),
});

export type Feature = z.infer<typeof FeatureSchema>;

export type Token = {
  token: Uint8Array;
  featureAct: number;
};
