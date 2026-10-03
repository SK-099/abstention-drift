"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Home from "@/components/Home";
import NotebookView from "@/components/NotebookView";

function Router() {
  const nb = useSearchParams().get("nb");
  return nb ? <NotebookView notebookId={nb} /> : <Home />;
}

export default function Page() {
  return (
    <Suspense fallback={null}>
      <Router />
    </Suspense>
  );
}
