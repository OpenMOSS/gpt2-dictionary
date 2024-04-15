import { AppNavbar } from "@/components/app/navbar";
import { DictionaryCard } from "@/components/dictionary/dictionary-card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { DictionarySchema } from "@/types/dictionary";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useAsyncFn, useMount } from "react-use";
import { z } from "zod";

export const DictionaryPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const [dictionariesState, fetchDictionaries] = useAsyncFn(async () => {
    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries`)
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [selectedDictionary, setSelectedDictionary] = useState<string | null>(null);

  const [dictionaryState, fetchDictionary] = useAsyncFn(async (dictionary: string | null) => {
    if (!dictionary) {
      alert("Please select a dictionary first");
      return;
    }

    const feature = await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}`, {
      method: "GET",
      headers: {
        Accept: "application/x-msgpack",
      },
    })
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
        })
      )
      .then((res) => DictionarySchema.parse(res));

    setSearchParams({
      dictionary,
    });

    return feature;
  });

  useMount(async () => {
    await fetchDictionaries();
    if (searchParams.get("dictionary")) {
      setSelectedDictionary(searchParams.get("dictionary"));
      fetchDictionary(searchParams.get("dictionary"));
    }
  });

  return (
    <div>
      <AppNavbar />
      <div className="pt-4 pb-20 px-20 flex flex-col items-center gap-12">
        <div className="container grid grid-cols-[auto_600px_auto] justify-center items-center gap-4">
          <span className="font-bold justify-self-end">Select dictionary:</span>
          <Select
            disabled={dictionariesState.loading && dictionaryState.loading}
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
            disabled={dictionariesState.loading && dictionaryState.loading}
            onClick={async () => {
              await fetchDictionary(selectedDictionary);
            }}
          >
            Go
          </Button>
        </div>
        {dictionaryState.loading && (
          <div>
            Loading Dictionary <span className="font-bold">{selectedDictionary}</span>...
          </div>
        )}
        {dictionaryState.error && <div className="text-red-500 font-bold">Error: {dictionaryState.error.message}</div>}
        {!dictionaryState.loading && dictionaryState.value && <DictionaryCard dictionary={dictionaryState.value} />}
      </div>
    </div>
  );
};
