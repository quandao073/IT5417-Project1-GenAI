import type { Metadata } from "next";
import { CompareView } from "@/components/CompareView";

export const metadata: Metadata = { title: "Compare modes - CXR Semantic Retrieval" };

export default function ComparePage() {
  return <CompareView />;
}
