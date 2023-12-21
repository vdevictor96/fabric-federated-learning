/*
 TODO remove - temp file for testing
*/

import {Object, Property} from 'fabric-contract-api';

@Object()
export class Model {
   
    @Property()
    public ID: string;

    @Property()
    public Size: number;

    @Property()
    public Owner: string;

}
