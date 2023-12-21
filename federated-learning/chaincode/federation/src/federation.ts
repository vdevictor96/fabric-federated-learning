import {Context, Contract, Info, Returns, Transaction} from 'fabric-contract-api';
import stringify from 'json-stringify-deterministic';
import sortKeysRecursive from 'sort-keys-recursive';
import {Model} from './model';

@Info({title: 'ModelTransfer', description: 'Smart contract for submitting models'})
export class ModelTransferContract extends Contract {

    @Transaction()
    public async InitLedger(ctx: Context): Promise<void> {
        const models: Model[] = [
            {
                ID: 'model1',
                Size: 5,
                Owner: 'Tomoko',
            },
            {
                ID: 'model2',
                Size: 5,
                Owner: 'Brad',
            },
            {
                ID: 'model3',
                Size: 10,
                Owner: 'Jin Soo',
            },
            {
                ID: 'model4',
                Size: 10,
                Owner: 'Max',
            },
        ];

        for (const model of models) {
            // example of how to write to world state deterministically
            // use convetion of alphabetic order
            // we insert data in alphabetic order using 'json-stringify-deterministic' and 'sort-keys-recursive'
            // when retrieving data, in any lang, the order of data will be the same and consequently also the corresonding hash
            await ctx.stub.putState(model.ID, Buffer.from(stringify(sortKeysRecursive(model))));
            console.info(`Model ${model.ID} initialized`);
        }
    }

    // CreateModel issues a new model to the world state with given details.
    @Transaction()
    public async CreateModel(ctx: Context, id: string, size: number, owner: string): Promise<void> {
        const exists = await this.ModelExists(ctx, id);
        if (exists) {
            throw new Error(`The model ${id} already exists`);
        }

        const model = {
            ID: id,
            Size: size,
            Owner: owner,
        };
        // we insert data in alphabetic order using 'json-stringify-deterministic' and 'sort-keys-recursive'
        await ctx.stub.putState(id, Buffer.from(stringify(sortKeysRecursive(model))));
    }

    // ReadModel returns the model stored in the world state with given id.
    @Transaction(false)
    public async ReadModel(ctx: Context, id: string): Promise<string> {
        const modelJSON = await ctx.stub.getState(id); // get the model from chaincode state
        if (!modelJSON || modelJSON.length === 0) {
            throw new Error(`The model ${id} does not exist`);
        }
        return modelJSON.toString();
    }

    // UpdateModel updates an existing model in the world state with provided parameters.
    @Transaction()
    public async UpdateModel(ctx: Context, id: string, size: number, owner: string): Promise<void> {
        const exists = await this.ModelExists(ctx, id);
        if (!exists) {
            throw new Error(`The model ${id} does not exist`);
        }

        // overwriting original model with new model
        const updatedModel = {
            ID: id,
            Size: size,
            Owner: owner,
        };
        // we insert data in alphabetic order using 'json-stringify-deterministic' and 'sort-keys-recursive'
        return ctx.stub.putState(id, Buffer.from(stringify(sortKeysRecursive(updatedModel))));
    }

    // DeleteModel deletes an given model from the world state.
    @Transaction()
    public async DeleteModel(ctx: Context, id: string): Promise<void> {
        const exists = await this.ModelExists(ctx, id);
        if (!exists) {
            throw new Error(`The model ${id} does not exist`);
        }
        return ctx.stub.deleteState(id);
    }

    // ModelExists returns true when model with given ID exists in world state.
    @Transaction(false)
    @Returns('boolean')
    public async ModelExists(ctx: Context, id: string): Promise<boolean> {
        const modelJSON = await ctx.stub.getState(id);
        return modelJSON && modelJSON.length > 0;
    }

    // TransferModel updates the owner field of model with given id in the world state, and returns the old owner.
    @Transaction()
    public async TransferModel(ctx: Context, id: string, newOwner: string): Promise<string> {
        const modelString = await this.ReadModel(ctx, id);
        const model = JSON.parse(modelString);
        const oldOwner = model.Owner;
        model.Owner = newOwner;
        // we insert data in alphabetic order using 'json-stringify-deterministic' and 'sort-keys-recursive'
        await ctx.stub.putState(id, Buffer.from(stringify(sortKeysRecursive(model))));
        return oldOwner;
    }

    // GetAllModels returns all models found in the world state.
    @Transaction(false)
    @Returns('string')
    public async GetAllModels(ctx: Context): Promise<string> {
        const allResults = [];
        // range query with empty string for startKey and endKey does an open-ended query of all models in the chaincode namespace.
        const iterator = await ctx.stub.getStateByRange('', '');
        let result = await iterator.next();
        while (!result.done) {
            const strValue = Buffer.from(result.value.value.toString()).toString('utf8');
            let record;
            try {
                record = JSON.parse(strValue);
            } catch (err) {
                console.log(err);
                record = strValue;
            }
            allResults.push(record);
            result = await iterator.next();
        }
        return JSON.stringify(allResults);
    }

}
