import { useState } from "react";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Dictionary, DictionarySample, DictionarySampleSchema } from "@/types/dictionary";
import Plot from "react-plotly.js";
import { useAsyncFn } from "react-use";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { Textarea } from "../ui/textarea";
import { DictionarySampleArea } from "./sample";

const DictionaryCustomInputArea = ({ dictionary }: { dictionary: Dictionary }) => {
  const [customInput, setCustomInput] = useState<string>("");
  const [samples, setSamples] = useState<DictionarySample[]>([]);
  const [state, submit] = useAsyncFn(async () => {
    if (!customInput) {
      alert("Please enter your input.");
      return;
    }
    const sample = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${
        dictionary.dictionaryName
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
      .then((res) => DictionarySampleSchema.parse(res));
    setSamples((prev) => [...prev, sample]);
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
      {samples.length > 0 && (
        <DictionarySampleArea
          samples={samples}
          dictionaryName={dictionary.dictionaryName}
          onSamplesChange={setSamples}
        />
      )}
    </div>
  );
};

export const DictionaryCard = ({ dictionary }: { dictionary: Dictionary }) => {
  const [showCustomInput, setShowCustomInput] = useState<boolean>(false);

  return (
    <Card className="container">
      <CardHeader>
        <CardTitle className="flex justify-between items-center text-xl">
          <span>
            #{dictionary.dictionaryName}{" "}
            <span className="font-medium">
              (Alive Feature Count = <span className="font-bold">{dictionary.aliveFeatureCount}</span>)
            </span>
          </span>
          <Button onClick={() => setShowCustomInput((prev) => !prev)}>
            {showCustomInput ? "Hide Custom Input" : "Try Custom Input"}
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          {showCustomInput && <DictionaryCustomInputArea dictionary={dictionary} />}

          <div className="flex flex-col w-full gap-4">
            <p className="font-bold">Activation Times Histogram</p>
            <Plot
              data={dictionary.featureActivationTimesHistogram}
              layout={{
                xaxis: { title: "Log Feature Activation Times" },
                yaxis: { title: "Count" },
                margin: { t: 0, b: 40 },
                showlegend: false,
              }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
