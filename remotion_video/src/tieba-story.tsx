import React, {useMemo} from "react";
import {
  AbsoluteFill,
  Audio,
  Img,
  Sequence,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import type {TiebaStoryProps} from "./types";

const FadeInOut: React.FC<{children: React.ReactNode; total: number; edge: number}> = ({children, total, edge}) => {
  const frame = useCurrentFrame();
  const alphaIn = interpolate(frame, [0, edge], [0, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});
  const alphaOut = interpolate(frame, [total - edge, total], [1, 0], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});
  const opacity = Math.min(alphaIn, alphaOut);
  return <div style={{opacity}}>{children}</div>;
};

const splitSubtitle = (text: string): string[] => {
  const cleaned = text
    .replace(/\s+/g, "")
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .trim();
  if (!cleaned) return [];
  const raw = cleaned.split(/(?<=[。！？!?])|(?<=，)|(?<=,)/g).map((s) => s.trim()).filter(Boolean);
  const out: string[] = [];
  for (const part of raw.length ? raw : [cleaned]) {
    if (part.length <= 26) {
      out.push(part);
      continue;
    }
    let buf = "";
    for (const ch of part) {
      buf += ch;
      if (buf.length >= 22) {
        out.push(buf);
        buf = "";
      }
    }
    if (buf) out.push(buf);
  }
  return out;
};

const subtitleDisplay = (text: string): string => {
  return text.replace(/[，,。.]/g, "").trim();
};

const Subtitles: React.FC<{text?: string; total: number}> = ({text, total}) => {
  if (!text || !text.trim()) return null;
  const frame = useCurrentFrame();
  const chunks = useMemo(() => splitSubtitle(text), [text]);
  if (chunks.length === 0) return null;
  const per = Math.max(1, Math.floor(total / Math.max(2, chunks.length + 1)));
  const idx = Math.min(chunks.length - 1, Math.max(0, Math.floor(frame / per)));
  const line1 = subtitleDisplay(chunks[idx] ?? "");
  if (!line1) return null;
  return (
    <AbsoluteFill
      style={{
        justifyContent: "flex-end",
        alignItems: "center",
        paddingBottom: "6%",
        pointerEvents: "none",
      }}
    >
      <FadeInOut total={total} edge={12}>
        <div
          style={{
            backgroundColor: "rgba(0,0,0,0.6)",
            borderRadius: 12,
            color: "#FAF6F0",
            fontFamily: "sans-serif",
            fontSize: 42,
            lineHeight: 1.3,
            padding: "14px 24px",
            maxWidth: "90%",
            textAlign: "center",
          }}
        >
          <div style={{whiteSpace: "nowrap"}}>{line1}</div>
        </div>
      </FadeInOut>
    </AbsoluteFill>
  );
};

const KenBurnsImage: React.FC<{src: string; total: number; seed: number}> = ({src, total, seed}) => {
  const frame = useCurrentFrame();
  const scale = interpolate(frame, [0, total], [1, 1.06], {extrapolateRight: "clamp"});
  const x = interpolate(frame, [0, total], [0, (seed % 2 === 0 ? -1 : 1) * 18], {extrapolateRight: "clamp"});
  const y = interpolate(frame, [0, total], [0, (seed % 3 === 0 ? -1 : 1) * 10], {extrapolateRight: "clamp"});
  return (
    <AbsoluteFill style={{transform: `translate3d(${x}px, ${y}px, 0) scale(${scale})`}}>
      <Img
        src={staticFile(src)}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
        }}
      />
    </AbsoluteFill>
  );
};

export const TiebaStory: React.FC<TiebaStoryProps> = ({scenes, bgm}) => {
  const {fps} = useVideoConfig();
  const frame = useCurrentFrame();

  const sceneFrames = useMemo(() => {
    let from = 0;
    return scenes.map((s) => {
      const dur = Math.max(1, Math.round((s.duration_s || 0) * fps));
      const start = from;
      from += dur;
      return {scene: s, from: start, durationInFrames: dur};
    });
  }, [fps, scenes]);

  const active = useMemo(() => {
    for (let i = 0; i < sceneFrames.length; i++) {
      const s = sceneFrames[i];
      if (frame >= s.from && frame < s.from + s.durationInFrames) {
        return s;
      }
    }
    return null;
  }, [frame, sceneFrames]);

  const bgmVolume = useMemo(() => {
    if (!bgm?.src) return 0;
    if (!active) return bgm.volume;
    const local = frame - active.from;
    const duckIn = interpolate(local, [0, 10], [1, 0.55], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});
    const duckOut = interpolate(local, [active.durationInFrames - 10, active.durationInFrames], [0.55, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
    const duck = Math.min(duckIn, duckOut);
    return bgm.volume * duck;
  }, [active, bgm?.src, bgm?.volume, frame]);

  return (
    <AbsoluteFill style={{backgroundColor: "black"}}>
      {bgm?.src ? <Audio src={staticFile(bgm.src)} volume={bgmVolume} /> : null}
      {sceneFrames.map(({scene, from, durationInFrames: dur}) => {
        return (
          <Sequence key={scene.scene_id} from={from} durationInFrames={dur}>
            <AbsoluteFill>
              <KenBurnsImage src={scene.image} total={dur} seed={scene.scene_id} />
              <AbsoluteFill
                style={{
                  background:
                    "radial-gradient(1200px 600px at 50% 35%, rgba(250, 246, 240, 0.08), rgba(0,0,0,0.55))",
                  mixBlendMode: "multiply",
                }}
              />
              <Audio src={staticFile(scene.audio)} volume={1} />
              <Subtitles text={scene.subtitle} total={dur} />
            </AbsoluteFill>
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
