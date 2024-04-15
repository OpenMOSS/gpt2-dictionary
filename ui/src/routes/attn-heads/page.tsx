import { AppNavbar } from "@/components/app/navbar";
import { AttentionHeadCard } from "@/components/attn-head/attn-head-card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AttentionHead, AttentionHeadSchema } from "@/types/attn-head";
import camelcaseKeys from "camelcase-keys";
import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useAsyncFn, useMount } from "react-use";

export const AttentionHeadPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const [selectedLayer, setSelectedLayer] = useState<number>(0);

  const [attnHeads, setAttnHeads] = useState<AttentionHead[]>([]);

  const [attnHeadState, fetchAttnHead] = useAsyncFn(async (layer: number) => {
    setSearchParams({
      layer: layer.toString(),
    });
    setAttnHeads([]);
    for (let i = 0; i < 12; i++) {
      const attnHead = await fetch(`${import.meta.env.VITE_BACKEND_URL}/attn_heads/${layer}/${i}`, {
        method: "GET",
      })
        .then(async (res) => {
          if (!res.ok) {
            throw new Error(await res.text());
          }
          return res.json();
        })
        .then((res) =>
          camelcaseKeys(res, {
            deep: true,
          })
        )
        .then((res) => AttentionHeadSchema.parse(res));

      setAttnHeads((heads) => {
        if (heads.some((head) => head.head === attnHead.head)) {
          return heads;
        }
        return [...heads, attnHead];
      });
    }
  });

  useMount(async () => {
    if (searchParams.get("layer")) {
      setSelectedLayer(parseInt(searchParams.get("layer")!));
      await fetchAttnHead(parseInt(searchParams.get("layer")!));
    }
  });

  return (
    <div>
      <AppNavbar />
      <div className="pt-4 pb-20 px-20 flex flex-col items-center gap-12">
        <div className="container grid grid-cols-[auto_600px_auto] justify-center items-center gap-4">
          <span className="font-bold justify-self-end">Select layer:</span>
          <Select
            disabled={attnHeadState.loading}
            value={selectedLayer.toString()}
            onValueChange={(value) => {
              setSelectedLayer(parseInt(value));
            }}
          >
            <SelectTrigger className="bg-white">
              <SelectValue placeholder="Select a dictionary" />
            </SelectTrigger>
            <SelectContent>
              {new Array(12).fill(null).map((_, i) => (
                <SelectItem key={i} value={i.toString()}>
                  {i}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            disabled={attnHeadState.loading}
            onClick={async () => {
              await fetchAttnHead(selectedLayer);
            }}
          >
            Go
          </Button>
        </div>
        {attnHeadState.loading && (
          <div>
            Loading Attention Head of Layer <span className="font-bold">{selectedLayer}</span>...
          </div>
        )}
        {attnHeadState.error && <div className="text-red-500 font-bold">Error: {attnHeadState.error.message}</div>}
        {!attnHeadState.loading &&
          attnHeads.map((attnHead, idx) => <AttentionHeadCard key={idx} attnHead={attnHead} />)}
      </div>
    </div>
  );
};
