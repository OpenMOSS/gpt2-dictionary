import { AppNavbar } from "@/components/app/navbar";
import { FeatureCard } from "@/components/feature/feature-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FeatureSchema } from "@/types/feature";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useAsyncFn, useMount } from "react-use";
import { z } from "zod";

export const FeaturesPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const [dictionariesState, fetchDictionaries] = useAsyncFn(async () => {
    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries`)
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [selectedDictionary, setSelectedDictionary] = useState<string | null>(null);

  const [featureIndex, setFeatureIndex] = useState<number>(0);
  const [loadingRandomFeature, setLoadingRandomFeature] = useState<boolean>(false);

  const [featureState, fetchFeature] = useAsyncFn(
    async (dictionary: string | null, featureIndex: number | string = "random") => {
      if (!dictionary) {
        alert("Please select a dictionary first");
        return;
      }

      setLoadingRandomFeature(featureIndex === "random");

      const feature = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}`,
        {
          method: "GET",
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
            stopPaths: ["sample_groups.samples.context"],
          })
        )
        .then((res) => FeatureSchema.parse(res));
      setFeatureIndex(feature.featureIndex);
      setSearchParams({
        dictionary,
        featureIndex: feature.featureIndex.toString(),
      });
      return feature;
    }
  );

  useMount(async () => {
    await fetchDictionaries();
    if (searchParams.get("dictionary")) {
      setSelectedDictionary(searchParams.get("dictionary"));
    }
    if (searchParams.get("featureIndex")) {
      setFeatureIndex(parseInt(searchParams.get("featureIndex")!));
      fetchFeature(searchParams.get("dictionary"), searchParams.get("featureIndex")!);
    }
  });

  useEffect(() => {
    if (dictionariesState.value && selectedDictionary === null) {
      setSelectedDictionary(dictionariesState.value[0]);
      fetchFeature(dictionariesState.value[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dictionariesState.value]);

  return (
    <div>
      <AppNavbar />
      <div className="pt-4 pb-20 px-20 flex flex-col items-center gap-12">
        <div className="container grid grid-cols-[auto_600px_auto_auto] justify-center items-center gap-4">
          <span className="font-bold justify-self-end">Select dictionary:</span>
          <Select
            disabled={dictionariesState.loading || featureState.loading}
            value={selectedDictionary || undefined}
            onValueChange={setSelectedDictionary}
          >
            <SelectTrigger className="bg-white">
              <SelectValue placeholder="Select a dictionary" />
            </SelectTrigger>
            <SelectContent>
              {dictionariesState.value?.map((dictionary, i) => (
                <SelectItem key={i} value={dictionary}>
                  {dictionary}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            disabled={dictionariesState.loading || featureState.loading}
            onClick={async () => {
              await fetchFeature(selectedDictionary);
            }}
          >
            Go
          </Button>
          <span className="font-bold"></span>
          <span className="font-bold justify-self-end">Choose a specific feature:</span>
          <Input
            disabled={dictionariesState.loading || selectedDictionary === null || featureState.loading}
            id="feature-input"
            className="bg-white"
            type="number"
            value={featureIndex.toString()}
            onChange={(e) => setFeatureIndex(parseInt(e.target.value))}
          />
          <Button
            disabled={dictionariesState.loading || selectedDictionary === null || featureState.loading}
            onClick={async () => await fetchFeature(selectedDictionary, featureIndex)}
          >
            Go
          </Button>
          <Button
            disabled={dictionariesState.loading || selectedDictionary === null || featureState.loading}
            onClick={async () => {
              await fetchFeature(selectedDictionary);
            }}
          >
            Show Random Feature
          </Button>
        </div>
        {featureState.loading && !loadingRandomFeature && (
          <div>
            Loading Feature <span className="font-bold">#{featureIndex}</span>...
          </div>
        )}
        {featureState.loading && loadingRandomFeature && <div>Loading Random Living Feature...</div>}
        {featureState.error && <div className="text-red-500 font-bold">Error: {featureState.error.message}</div>}
        {!featureState.loading && featureState.value && <FeatureCard feature={featureState.value} />}
      </div>
    </div>
  );
};
