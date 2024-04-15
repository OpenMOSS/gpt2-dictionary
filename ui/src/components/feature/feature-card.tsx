import { Feature, SampleSchema } from "@/types/feature";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useState } from "react";
import Plot from "react-plotly.js";
import { useAsyncFn } from "react-use";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { Textarea } from "../ui/textarea";
import { FeatureInterpretation } from "./interpret";
import { FeatureActivationSample, FeatureSampleGroup } from "./sample";

const FeatureCustomInputArea = ({ feature }: { feature: Feature }) => {
  const [customInput, setCustomInput] = useState<string>("");
  const [state, submit] = useAsyncFn(async () => {
    if (!customInput) {
      alert("Please enter your input.");
      return;
    }
    return await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${feature.dictionaryName}/features/${
        feature.featureIndex
      }/custom?input_text=${encodeURIComponent(customInput)}`,
      {
        method: "POST",
        headers: {
          Accept: "application/x-msgpack",
        },
      }
    )
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.arrayBuffer())
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .then((res) => decode(new Uint8Array(res)) as any)
      .then((res) =>
        camelcaseKeys(res, {
          deep: true,
          stopPaths: ["context"],
        })
      )
      .then((res) => SampleSchema.parse(res));
  }, [customInput]);

  return (
    <div className="flex flex-col gap-4">
      <p className="font-bold">Custom Input</p>
      <Textarea
        placeholder="Type your custom input here."
        value={customInput}
        onChange={(e) => setCustomInput(e.target.value)}
      />
      <Button onClick={submit} disabled={state.loading}>
        Submit
      </Button>
      {state.error && <p className="text-red-500">{state.error.message}</p>}
      {state.value && (
        <>
          <FeatureActivationSample
            sample={state.value}
            sampleName="Custom Input"
            maxFeatureAct={feature.maxFeatureAct}
          />
          <p className="font-bold">Custom Input Max Activation: {Math.max(...state.value.featureActs).toFixed(3)}</p>
        </>
      )}
    </div>
  );
};

export const FeatureCard = ({ feature }: { feature: Feature }) => {
  const analysisNameMap = (analysisName: string) => {
    if (analysisName === "top_activations") {
      return "Top Activations";
    } else if (/^subsample-/.test(analysisName)) {
      const [, proportion] = analysisName.split("-");
      const percentage = parseFloat(proportion) * 100;
      return `Subsample ${percentage}%`;
    }
  };

  const [showCustomInput, setShowCustomInput] = useState<boolean>(false);

  return (
    <Card className="container">
      <CardHeader>
        <CardTitle className="flex justify-between items-center text-xl">
          <span>
            #{feature.featureIndex}{" "}
            <span className="font-medium">
              (Activation Times = <span className="font-bold">{feature.actTimes}</span>)
            </span>
          </span>
          <Button onClick={() => setShowCustomInput((prev) => !prev)}>
            {showCustomInput ? "Hide Custom Input" : "Try Custom Input"}
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          {showCustomInput && <FeatureCustomInputArea feature={feature} />}

          <FeatureInterpretation feature={feature} />

          <div className="flex flex-col w-full gap-4">
            <p className="font-bold">Activation Histogram</p>
            <Plot
              data={feature.featureActivationHistogram}
              layout={{
                xaxis: { title: "Activation" },
                yaxis: { title: "Count" },
                bargap: 0.2,
                margin: { t: 0, b: 40 },
                showlegend: false,
              }}
            />
          </div>

          {feature.logits && (
            <div className="flex flex-col w-full gap-4">
              <p className="font-bold">Logits</p>
              <div className="flex gap-4">
                <div className="flex flex-col w-1/2 gap-4">
                  <p className="font-bold">Top Positive</p>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Token</TableHead>
                        <TableHead>Logit</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {feature.logits.topPositive.map((token) => (
                        <TableRow key={token.token}>
                          <TableCell className="underline whitespace-pre-wrap decoration-slate-400 decoration-1 decoration-dotted underline-offset-[6px]">
                            {token.token}
                          </TableCell>
                          <TableCell>{token.logit.toFixed(3)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                <div className="flex flex-col w-1/2 gap-4">
                  <p className="font-bold">Top Negative</p>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Token</TableHead>
                        <TableHead>Logit</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {feature.logits.topNegative.map((token) => (
                        <TableRow key={token.token}>
                          <TableCell className="underline whitespace-pre-wrap decoration-slate-400 decoration-1 decoration-dotted underline-offset-[6px]">
                            {token.token}
                          </TableCell>
                          <TableCell>{token.logit.toFixed(3)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
              <Plot
                data={feature.logits.histogram}
                layout={{
                  bargap: 0.2,
                  margin: { t: 0, b: 40 },
                  showlegend: false,
                }}
              />
            </div>
          )}

          <div className="flex flex-col w-full gap-4">
            <Tabs defaultValue="top_activations">
              <TabsList className="font-bold">
                {feature.sampleGroups.map((sampleGroup) => (
                  <TabsTrigger key={`tab-trigger-${sampleGroup.analysisName}`} value={sampleGroup.analysisName}>
                    {analysisNameMap(sampleGroup.analysisName)}
                  </TabsTrigger>
                ))}
              </TabsList>
              {feature.sampleGroups.map((sampleGroup) => (
                <TabsContent
                  key={`tab-content-${sampleGroup.analysisName}`}
                  value={sampleGroup.analysisName}
                  className="mt-0"
                >
                  <FeatureSampleGroup feature={feature} sampleGroup={sampleGroup} />
                </TabsContent>
              ))}
            </Tabs>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
