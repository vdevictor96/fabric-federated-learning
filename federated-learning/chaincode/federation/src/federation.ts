import {
  Context,
  Contract,
  Info,
  Returns,
  Transaction,
} from "fabric-contract-api";
import stringify from "json-stringify-deterministic";
import sortKeysRecursive from "sort-keys-recursive";
import { Model, ModelParams, ModelWeights } from "./model";

@Info({
  title: "ModelTransfer",
  description: "Smart contract for submitting models",
})
export class ModelTransferContract extends Contract {
  static readonly UTF8_DECODER = new TextDecoder();

  @Transaction()
  public async InitLedger(ctx: Context): Promise<void> {
    const models: Model[] = [
      {
        id: "model1",
        modelParams: "",
        owner: "Tomoko",
      },
      {
        id: "model2",
        modelParams: "",
        owner: "Brad",
      },
      {
        id: "model3",
        modelParams: "",
        owner: "Jin Soo",
      },
      {
        id: "model4",
        modelParams: "",
        owner: "Max",
      },
    ];

    for (const model of models) {
      // example of how to write to world state deterministically
      // use convetion of alphabetic order
      // we insert data in alphabetic order using 'json-stringify-deterministic' and 'sort-keys-recursive'
      // when retrieving data, in any lang, the order of data will be the same and consequently also the corresonding hash
      await ctx.stub.putState(
        model.id,
        Buffer.from(stringify(sortKeysRecursive(model)))
      );
      console.info(`Model ${model.id} initialized`);
    }
  }

  // CreateModel issues a new model to the world state with given details.
  @Transaction()
  public async CreateModel(
    ctx: Context,
    id: string,
    modelParams: string,
    owner: string
  ): Promise<void> {
    // const clientIdentity = new ClientIdentity(ctx.stub);
    // const clientId = clientIdentity.getID();
    const exists = await this.ModelExists(ctx, id);
    if (exists) {
      throw new Error(`The model ${id} already exists`);
    }

    const model = {
      id,
      modelParams,
      owner,
    };
    // we insert data in alphabetic order using 'json-stringify-deterministic' and 'sort-keys-recursive'
    await ctx.stub.putState(
      id,
      Buffer.from(stringify(sortKeysRecursive(model)))
    );
  }

  // CreateModel issues a new model to the world state with given details.
  @Transaction()
  public async SubmitLocalModel(
    ctx: Context,
    id: string,
    jsonParams: string
  ): Promise<void> {
    // console.log(jsonParams);
    // const clientIdentity = new ClientIdentity(ctx.stub);
    // const clientId = clientIdentity.getID();

    const exists = await this.ModelExists(ctx, id);
    if (exists) {
      throw new Error(`The model ${id} already exists`);
    }

    const modelParams: ModelParams = JSON.parse(jsonParams);
    console.log(modelParams);

    // for (const [key, value] of Object.entries(modelParams)) {
    //     console.log(key);
    //     console.log(value);
    //     console.log(value.length);
    // }
    await this.CreateModel(ctx, id, jsonParams, "Victor");
    // return this.ReadModel(ctx, newModelId);
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

  @Transaction()
  public async AggregateModels(
    ctx: Context,
    jsonModelIds: string
  ): Promise<void> {
    const modelWeights: ModelParams[] = [];
    const aggregatedWeights: ModelParams = {};
    const modelIds = JSON.parse(jsonModelIds);
    // fill up model JSONs with all models
    // fill up modelWeights with all models' weights
    for (const modelId of modelIds) {
      const resultBytes = await ctx.stub.getState(modelId);
      const resultJson = ModelTransferContract.UTF8_DECODER.decode(resultBytes);
      const modelJSON = JSON.parse(resultJson);

      // fill up modelWeights with all models' weights
      const modelParams: ModelParams = JSON.parse(modelJSON.modelParams);

      modelWeights.push(modelParams);
    }

    // aggregate the weights
    for (const key in modelWeights[0]) {
      aggregatedWeights[key] = this.aggregateWeights(
        modelWeights.map((model) => model[key])
      );
    }

    // save it as a new model using this.CreateModel
    // convert aggregated weights to JSON and return
    // Convert aggregated weights to JSON
    const jsonParams = JSON.stringify(aggregatedWeights);

    const newModelId: string = modelIds[0] + "and" + modelIds[1];

    // Save the new aggregated model
    await this.CreateModel(ctx, newModelId, jsonParams, "Victor");

    // return this.ReadModel(ctx, newModelId);
  }

  private aggregateWeights(modelWeights: ModelWeights[]): ModelWeights {
    if (this.isArrayOfNumbers(modelWeights)) {
      // Base case: array of numbers
      return (
        modelWeights.reduce((sum, num) => Number(sum) + Number(num), 0) /
        modelWeights.length
      );
    } else {
      // Recursive case: array of arrays
      const length = (modelWeights[0] as ModelWeights[]).length;
      const aggregated = new Array(length)
        .fill(null)
        .map((_, i) =>
          this.aggregateWeights(
            modelWeights.map((array) => (array as ModelWeights[])[i])
          )
        );
      return aggregated;
    }
  }

  private isArrayOfNumbers(array: ModelWeights[]): array is number[] {
    return array.every((element) => typeof element === "number");
  }

  // UpdateModel updates an existing model in the world state with provided parameters.
  @Transaction()
  public async UpdateModel(
    ctx: Context,
    id: string,
    size: number,
    owner: string
  ): Promise<void> {
    const exists = await this.ModelExists(ctx, id);
    if (!exists) {
      throw new Error(`The model ${id} does not exist`);
    }

    // overwriting original model with new model
    const updatedModel = {
      id: id,
      Size: size,
      owner: owner,
    };
    // we insert data in alphabetic order using 'json-stringify-deterministic' and 'sort-keys-recursive'
    return ctx.stub.putState(
      id,
      Buffer.from(stringify(sortKeysRecursive(updatedModel)))
    );
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
  @Returns("boolean")
  public async ModelExists(ctx: Context, id: string): Promise<boolean> {
    const modelJSON = await ctx.stub.getState(id);
    return modelJSON && modelJSON.length > 0;
  }

  // TransferModel updates the owner field of model with given id in the world state, and returns the old owner.
  @Transaction()
  public async TransferModel(
    ctx: Context,
    id: string,
    newOwner: string
  ): Promise<string> {
    const modelString = await this.ReadModel(ctx, id);
    const model = JSON.parse(modelString);
    const oldOwner = model.owner;
    model.owner = newOwner;
    // we insert data in alphabetic order using 'json-stringify-deterministic' and 'sort-keys-recursive'
    await ctx.stub.putState(
      id,
      Buffer.from(stringify(sortKeysRecursive(model)))
    );
    return oldOwner;
  }

  // GetAllModels returns all models found in the world state.
  @Transaction(false)
  @Returns("string")
  public async GetAllModels(ctx: Context): Promise<string> {
    const allResults = [];
    // range query with empty string for startKey and endKey does an open-ended query of all models in the chaincode namespace.
    const iterator = await ctx.stub.getStateByRange("", "");
    let result = await iterator.next();
    while (!result.done) {
      const strValue = Buffer.from(result.value.value.toString()).toString(
        "utf8"
      );
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
