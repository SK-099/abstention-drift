import { AudioLines, BookOpenCheck, FileText, HelpCircle, Layers, ListChecks, Network, Timer } from "lucide-react";
import type { ArtifactKind } from "@/lib/api";

export const KIND_META: Record<
  ArtifactKind,
  { label: string; icon: typeof FileText; tint: string; hint: string }
> = {
  audio: { label: "Audio overview", icon: AudioLines, tint: "bg-violet-500/15 text-violet-600", hint: "Two hosts discuss your sources" },
  briefing: { label: "Briefing doc", icon: FileText, tint: "bg-sky-500/15 text-sky-600", hint: "Summary, key points, open questions" },
  study_guide: { label: "Study guide", icon: BookOpenCheck, tint: "bg-emerald-500/15 text-emerald-600", hint: "Concepts, questions and answer key" },
  faq: { label: "FAQ", icon: HelpCircle, tint: "bg-amber-500/15 text-amber-600", hint: "Questions people would ask" },
  timeline: { label: "Timeline", icon: Timer, tint: "bg-rose-500/15 text-rose-600", hint: "Dated events in order" },
  mindmap: { label: "Mind map", icon: Network, tint: "bg-indigo-500/15 text-indigo-600", hint: "Topics and how they connect" },
  flashcards: { label: "Flashcards", icon: Layers, tint: "bg-teal-500/15 text-teal-600", hint: "Flip cards to study" },
  quiz: { label: "Quiz", icon: ListChecks, tint: "bg-orange-500/15 text-orange-600", hint: "Multiple choice with answers" },
};

export const KIND_ORDER: ArtifactKind[] = ["audio", "briefing", "study_guide", "faq", "timeline", "mindmap", "flashcards", "quiz"];
