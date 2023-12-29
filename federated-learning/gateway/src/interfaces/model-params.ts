/*
 * Interface for model weights, which are n dimensional arrays of numbers
 */
export type ModelWeights = number | ModelWeights[];

export interface ModelParams {
  [layer: string]: ModelWeights;
}
