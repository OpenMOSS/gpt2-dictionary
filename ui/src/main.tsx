import React from "react";
import ReactDOM from "react-dom/client";
import "./globals.css";
import { RouterProvider, createBrowserRouter } from "react-router-dom";
import { FeaturesPage } from "@/routes/features/page";
import { RootPage } from "./routes/page";
import { AttentionHeadPage } from "./routes/attn-heads/page";
import { DictionaryPage } from "./routes/dictionaries/page";

const router = createBrowserRouter([
  {
    path: "/features",
    element: <FeaturesPage />,
  },
  {
    path: "/dictionaries",
    element: <DictionaryPage />,
  },
  {
    path: "/attn-heads",
    element: <AttentionHeadPage />,
  },
  {
    path: "/",
    element: <RootPage />,
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
