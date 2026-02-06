export type TiebaScene = {
  scene_id: number;
  image: string;
  audio: string;
  duration_s: number;
  subtitle?: string;
};

export type TiebaStoryProps = {
  scenes: TiebaScene[];
  bgm?: {
    src: string;
    volume: number;
  };
};
