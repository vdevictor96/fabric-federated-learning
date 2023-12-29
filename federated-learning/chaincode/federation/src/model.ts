
import {Object, Property} from 'fabric-contract-api';

@Object()
export class Model {
   
    @Property()
    public id: string;

    @Property()
    public modelParams: string; // Base64 encoded string of the .pt file

    @Property()
    public owner: string;

}


/*
 * Interface for model weights, which are n dimensional arrays of numbers
 */
export type ModelWeights = number | ModelWeights[];

export interface ModelParams {
  [layer: string]: ModelWeights;
}
  
