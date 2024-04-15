import { AttentionHead } from "@/types/attn-head";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../ui/table";
import { FeatureLinkWithPreview } from "../app/feature-preview";

export const AttentionHeadCard = ({ attnHead }: { attnHead: AttentionHead }) => {
  return (
    <Card className="container">
      <CardHeader>
        <CardTitle className="text-xl">
          Attention Head {attnHead.layer}.{attnHead.head}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-12">
          {attnHead.attnScores.map((attnScoreGroup, idx) => (
            <div key={idx} className="flex flex-col gap-4">
              <p className="font-bold">
                {attnScoreGroup.dictionary1Name} {" -> "}
                {attnScoreGroup.dictionary2Name}
              </p>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Feature</TableHead>
                    <TableHead>Feature Attended</TableHead>
                    <TableHead>Attention Score</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {attnScoreGroup.topAttnScores.map((attnScore, idx) => (
                    <TableRow key={idx}>
                      <TableCell>
                        <FeatureLinkWithPreview
                          dictionaryName={attnScoreGroup.dictionary1Name}
                          featureIndex={attnScore.feature1Index}
                        />
                      </TableCell>
                      <TableCell>
                        <FeatureLinkWithPreview
                          dictionaryName={attnScoreGroup.dictionary2Name}
                          featureIndex={attnScore.feature2Index}
                        />
                      </TableCell>
                      <TableCell>{attnScore.attnScore.toFixed(3)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
