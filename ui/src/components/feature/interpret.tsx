import { Feature, Interpretation, InterpretationSchema } from "@/types/feature";
import { useState } from "react";
import { Button } from "../ui/button";
import { Ban, Check, Info } from "lucide-react";
import { useAsyncFn } from "react-use";
import { Textarea } from "../ui/textarea";
import camelcaseKeys from "camelcase-keys";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../ui/hover-card";

const FeatureCustomInterpretionArea = ({
  feature,
  defaultInterpretation,
  onInterpretation,
}: {
  feature: Feature;
  defaultInterpretation: Interpretation | null;
  onInterpretation: (interpretation: Interpretation) => void;
}) => {
  const [customInput, setCustomInput] = useState<string>(defaultInterpretation?.text || "");
  const [state, submit] = useAsyncFn(async () => {
    if (!customInput) {
      alert("Please enter your interpretation.");
      return;
    }
    const interpretation = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${feature.dictionaryName}/features/${
        feature.featureIndex
      }/interpret?type=custom&custom_interpretation=${encodeURIComponent(customInput)}`,
      {
        method: "POST",
      }
    )
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.json())
      .then((res) => InterpretationSchema.parse(camelcaseKeys(res)));
    onInterpretation(interpretation);
    return interpretation;
  }, [customInput]);

  return (
    <div className="flex flex-col gap-4">
      <Textarea
        placeholder="Type your custom interpretation here."
        value={customInput}
        onChange={(e) => setCustomInput(e.target.value)}
      />
      <Button onClick={submit} disabled={state.loading}>
        Submit
      </Button>
      {state.error && <p className="text-red-500">{state.error.message}</p>}
    </div>
  );
};

export const FeatureInterpretation = ({ feature }: { feature: Feature }) => {
  const [showCustomInput, setShowCustomInput] = useState<boolean>(false);

  const [interpretation, setInterpretation] = useState<Interpretation | null>(feature.interpretation);

  const [validating, setValidating] = useState<boolean>(false);

  const testNameMap = (method: string, passed: boolean) => {
    switch (method) {
      case "activation":
        return `Activation Prediction Test ${passed ? "Passed" : "Failed"}`;
      case "generative":
        return `Generative Test ${passed ? "Passed" : "Failed"}`;
      case "manual":
        return "Human Validated";
      default:
        return `Unknown Test ${method} ${passed ? "Passed" : "Failed"}`;
    }
  };

  const [state, autoInterpretation] = useAsyncFn(async () => {
    const interpretation = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${
        feature.dictionaryName
      }/features/${feature.featureIndex}/interpret?type=auto`,
      {
        method: "POST",
      }
    )
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.json())
      .then((res) => InterpretationSchema.parse(camelcaseKeys(res)));
    setInterpretation(interpretation);
    setValidating(true);
    const interpretationValidated = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${
        feature.dictionaryName
      }/features/${feature.featureIndex}/interpret?type=validate`,
      {
        method: "POST",
      }
    )
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.json())
      .then((res) => InterpretationSchema.parse(camelcaseKeys(res)))
      .finally(() => setValidating(false));
    setInterpretation(interpretationValidated);
    return interpretationValidated;
  });

  return (
    <div className="flex flex-col w-full gap-4">
      <p className="font-bold">Interpretation</p>
      <div className="flex gap-12">
        <div className="flex flex-col justify-between gap-4 basis-2/3 min-w-2/3">
          {!interpretation && <p className="text-neutral-500">No interpretation available.</p>}
          {interpretation && (
            <div className="flex gap-4">
              <p>
                {interpretation.text}
                {interpretation.detail && (
                  <HoverCard>
                    <HoverCardTrigger>
                      <Info size={20} className="hover:text-blue-500 inline-block mx-2" />
                    </HoverCardTrigger>
                    <HoverCardContent className="w-[800px] max-h-[400px] overflow-y-auto">
                      <div className="grid gap-4 whitespace-pre-wrap grid-cols-[auto,1fr]">
                        <div className="text-sm font-bold">Prompt:</div>
                        <div className="text-sm">{interpretation.detail.prompt}</div>
                        <div className="text-sm font-bold">Response:</div>
                        <div className="text-sm">{interpretation.detail.response}</div>
                      </div>
                    </HoverCardContent>
                  </HoverCard>
                )}
              </p>
            </div>
          )}
          <div className="flex gap-4">
            <Button size="sm" onClick={autoInterpretation} disabled={state.loading}>
              Automatic Interpretation
            </Button>
            <Button variant="secondary" size="sm" onClick={() => setShowCustomInput((prev) => !prev)}>
              {showCustomInput
                ? "Hide Custom Interpretation"
                : interpretation
                  ? "Edit Interpretation"
                  : "Provide Custom Interpretation"}
            </Button>
          </div>
        </div>
        <div className="flex flex-col gap-4 basis-1/3 min-w-1/3">
          {interpretation?.validation.map((validation, i) => (
            <div key={i} className="flex items-center gap-2">
              {validation.passed ? (
                <Check size={20} className="text-green-500" />
              ) : (
                <Ban size={20} className="text-red-500" />
              )}
              <p>{testNameMap(validation.method, validation.passed)}</p>
              {validation.detail && (
                <HoverCard>
                  <HoverCardTrigger>
                    <Info size={20} className="hover:text-blue-500" />
                  </HoverCardTrigger>
                  <HoverCardContent className="w-[800px] flex flex-col gap-2">
                    {validation.method === "activation" && (
                      <>
                        <p className="text-sm">
                          Activation prediction test checks if the LLM can simulate a Sparse Auto Encoder to predict the
                          activation of the feature, given the description of the feature.
                        </p>
                        <p className="text-sm">
                          For simplicity, the LLM is only asked to predict the token with the highest activation. The
                          test is passed if the predicted token is the same as the actual token. White spaces are
                          ignored.
                        </p>
                      </>
                    )}
                    {validation.method === "generative" && (
                      <>
                        <p className="text-sm">
                          Generative test checks if the LLM can generate a sentence that any of the tokens in the
                          sentence can activate the feature.
                        </p>
                        <p className="text-sm">
                          The test is passed if the generated sentence contains at least one token that activates the
                          feature to a certain threshold (currently set to 1.0).
                        </p>
                      </>
                    )}
                    <div className="grid gap-2 whitespace-pre-wrap grid-cols-[auto,1fr]">
                      <div className="text-sm font-bold">Prompt:</div>
                      <div className="text-sm">{validation.detail.prompt}</div>
                      <div className="text-sm font-bold">Response:</div>
                      <div className="text-sm">{validation.detail.response}</div>
                    </div>
                  </HoverCardContent>
                </HoverCard>
              )}
            </div>
          ))}
          {validating && <p className="text-neutral-500">Validating...</p>}
        </div>
      </div>
      {showCustomInput && (
        <FeatureCustomInterpretionArea
          feature={feature}
          defaultInterpretation={interpretation}
          onInterpretation={(interpretation) => {
            setInterpretation(interpretation);
            setShowCustomInput(false);
          }}
        />
      )}
    </div>
  );
};
