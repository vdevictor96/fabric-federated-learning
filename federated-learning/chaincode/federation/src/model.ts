
import {Object, Property} from 'fabric-contract-api';

@Object()
export class Model {
   
    @Property()
    public ID: string;

    @Property()
    public ModelParams: string; // Base64 encoded string of the .pt file

    @Property()
    public Owner: string;

}


export interface ModelParams {
    [key: string]: number[][];
  }
  
