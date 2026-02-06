import React from "react";
import {Composition} from "remotion";
import {TiebaStory} from "./tieba-story";
import type {TiebaStoryProps} from "./types";

const fps = 30;
const width = 1664;
const height = 928;

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition<any, TiebaStoryProps>
        id="TiebaStory"
        component={TiebaStory}
        width={width}
        height={height}
        fps={fps}
        durationInFrames={fps}
        defaultProps={{
          scenes: [],
          bgm: {
            src: "",
            volume: 0.12,
          },
        }}
        calculateMetadata={({props}) => {
          const seconds = props.scenes.reduce((acc, s) => acc + (s.duration_s || 0), 0);
          const durationInFrames = Math.max(1, Math.ceil(seconds * fps));
          return {durationInFrames, fps, width, height};
        }}
      />
    </>
  );
};
