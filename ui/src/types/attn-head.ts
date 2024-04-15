import { z } from "zod";

export const AttentionHeadSchema = z.object({
  layer: z.number(),
  head: z.number(),
  attnScores: z.array(
    z.object({
      dictionary1Name: z.string(),
      dictionary2Name: z.string(),
      topAttnScores: z.array(
        z.object({
          feature1Index: z.number(),
          feature2Index: z.number(),
          attnScore: z.number(),
        })
      ),
    })
  ),
});

export type AttentionHead = z.infer<typeof AttentionHeadSchema>;
