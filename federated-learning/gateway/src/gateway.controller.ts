import * as zlib from 'zlib';
import { decode, encode } from '@msgpack/msgpack';
import {
  Body,
  Controller,
  Delete,
  Get,
  HttpStatus,
  Param,
  Post,
  Res,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { GatewayService } from './gateway.service';
import { FileInterceptor } from '@nestjs/platform-express';
import { promises as fs } from 'fs';
import * as path from 'path';
import { Response } from 'express';
import { ModelDto } from './dtos/model.dto'; // Adjust the import path as needed
import { ModelParams, ModelWeights } from './interfaces/model-params';

@Controller('gateway')
export class GatewayController {
  constructor(private readonly gatewayService: GatewayService) {}

  @Get('hello')
  getHello(): string {
    return this.gatewayService.getHello();
  }

  @Post('upload')
  @UseInterceptors(FileInterceptor('file'))
  async uploadFile(@UploadedFile() file: Express.Multer.File, @Res() response) {
    console.log(file);
    // For downloading the file, you can send it back in the response
    // Define the path where the file will be saved
    const savePath = path.join(process.cwd(), file.originalname);
    // Write the file to disk
    await fs.writeFile(savePath, file.buffer);

    // Optionally, send a response to the client
    response
      .status(200)
      .send({ message: 'File uploaded successfully', path: savePath });
  }

  /**
   * This type of transaction would typically only be run once by an application the first time it was started after its
   * initial deployment. A new version of the chaincode deployed later would likely not need to run an "init" function.
   */
  @Get('initLedger')
  public async initLedger(@Res() response: Response) {
    try {
      console.log(
        '\n--> Submit Transaction: InitLedger, function creates the initial set of models on the ledger',
      );
      const contract = await this.gatewayService.getContract();
      const res = await contract.submitTransaction('InitLedger');
      console.log(res);
      console.log('*** Transaction committed successfully');
      response
        .status(200)
        .json({ message: 'Ledger initialized successfully', data: res });
    } catch (error: any) {
      console.error('Error initializing ledger', error.message);
      response
        .status(400)
        .json({ message: 'Error initializing ledger', error: error.message });
    }
  }

  /**
   * Evaluate a transaction to query ledger state.
   */
  @Get('allModels')
  public async getAllModels(@Res() response: Response) {
    try {
      console.log(
        '\n--> Evaluate Transaction: GetAllModels, function returns all the current models on the ledger',
      );
      const contract = await this.gatewayService.getContract();
      const resultBytes = await contract.evaluateTransaction('GetAllModels');
      const resultJson = GatewayService.UTF8_DECODER.decode(resultBytes);
      const result = JSON.parse(resultJson);
      console.log('*** Result:', result);
      response
        .status(200)
        .json({ message: 'Models retrieved correctly', data: result });
    } catch (error: any) {
      console.error('Error retreiving models', error.message);
      response
        .status(400)
        .json({ message: 'Error retreiving models', error: error.message });
    }
  }

  /**
   * Evaluate a transaction to query ledger state.
   */
  @Get('allModelNames')
  public async getAllModelNames(@Res() response: Response) {
    try {
      console.log(
        '\n--> Evaluate Transaction: GetAllModelNames, function returns all the current model names on the ledger',
      );
      const contract = await this.gatewayService.getContract();
      const resultBytes =
        await contract.evaluateTransaction('GetAllModelNames');
      const resultJson = GatewayService.UTF8_DECODER.decode(resultBytes);
      const result = JSON.parse(resultJson);
      console.log('*** Result:', result);
      response
        .status(200)
        .json({ message: 'Models retrieved correctly', data: result });
    } catch (error: any) {
      console.error('Error retreiving models', error.message);
      response
        .status(400)
        .json({ message: 'Error retreiving models', error: error.message });
    }
  }

  /**
   * Evaluate a transaction to query ledger state.
   */
  @Get('model/:id')
  public async getModelById(
    @Param('id') modelId: string,
    @Res() response: Response,
  ) {
    try {
      console.log(
        '\n--> Evaluate Transaction: ReadModel, function returns model attributes',
      );

      const contract = await this.gatewayService.getContract();
      const resultBytes = await contract.evaluateTransaction(
        'ReadModel',
        modelId,
      );

      const resultJson = GatewayService.UTF8_DECODER.decode(resultBytes);
      const result = JSON.parse(resultJson);
      // console.log('*** Result:', result);

      response
        .status(200)
        .json({ message: 'Model retrieved correctly', data: result });
    } catch (error: any) {
      console.error('Error retreiving model', error.message);
      response
        .status(400)
        .json({ message: 'Error retreiving model', error: error.message });
    }
  }

  /**
   * Evaluate a transaction to query ledger state.
   */
  @Delete('allModels')
  public async deleteAllModels(@Res() response: Response) {
    try {
      console.log(
        '\n--> Evaluate Transaction: DeleteAllModels, function deletes all the current models on the ledger',
      );
      const contract = await this.gatewayService.getContract();
      const resultBytes = await contract.submitTransaction('DeleteAllModels');
      const resultJson = GatewayService.UTF8_DECODER.decode(resultBytes);
      const result = JSON.parse(resultJson);
      console.log('*** Result:', result);
      response
        .status(200)
        .json({ message: 'All models deleted correctly', data: result });
    } catch (error: any) {
      console.error('Error deleting all models', error.message);
      response
        .status(400)
        .json({ message: 'Error deleting all models', error: error.message });
    }
  }

  /**
   * Evaluate a transaction to query ledger state.
   */
  @Delete('model/:id')
  public async deleteModelById(
    @Param('id') modelId: string,
    @Res() response: Response,
  ) {
    try {
      console.log(
        '\n--> Evaluate Transaction: DeleteModel, function deletes model',
      );

      const contract = await this.gatewayService.getContract();
      await contract.submitTransaction('DeleteModel', modelId);

      console.log('*** Transaction committed successfully');
      response
        .status(HttpStatus.OK)
        .json({ message: 'Model deleted succesfully' });
    } catch (error: any) {
      console.error('Error deleting model', error.message);
      response
        .status(HttpStatus.BAD_REQUEST)
        .json({ message: 'Error deleting model', error: error.message });
    }
  }

  /**
   * Submit a transaction synchronously, blocking until it has been committed to the ledger.
   */
  @Post('model')
  public async createModel(
    @Body() modelDto: ModelDto,
    @Res() response: Response,
  ) {
    try {
      console.log(modelDto);
      console.log(
        '\n--> Submit Transaction: CreateModel, creates new model with ID, ModelParams, Owner',
      );
      const contract = await this.gatewayService.getContract();
      await contract.submitTransaction(
        'CreateModel',
        modelDto.id,
        modelDto.modelParams,
        modelDto.owner,
      );
      console.log('*** Transaction committed successfully');
      response
        .status(HttpStatus.OK)
        .json({ message: 'Model created succesfully' });
    } catch (error: any) {
      console.error('Error creating model', error.message);
      response
        .status(HttpStatus.BAD_REQUEST)
        .json({ message: 'Error creating model', error: error.message });
    }
  }

  /**
   * Submit a transaction synchronously, blocking until it has been committed to the ledger.
   */
  @Post('local-model/:id')
  public async submitLocalModel(
    @Param('id') modelId: string,
    @Body() encodedModelParams: string,
    @Res() response: Response,
  ) {
    try {
      // console.log(modelParams);

      // for (const [key, value] of Object.entries(modelParams)) {
      //   console.log(key);
      //   console.log(value);
      //   console.log(value.length);
      // }

      console.log(
        '\n--> Submit Transaction: SubmitLocalModel, submits local trained model',
      );
      const contract = await this.gatewayService.getContract();
      await contract.submitTransaction(
        'SubmitLocalModel',
        modelId,
        encodedModelParams,
      );
      console.log('*** Transaction committed successfully');
      response
        .status(HttpStatus.OK)
        .json({ message: 'Model submitted succesfully' });
    } catch (error: any) {
      console.error('Error submitting model', error.message);
      response
        .status(HttpStatus.BAD_REQUEST)
        .json({ message: 'Error submitting model', error: error.message });
    }
  }

  @Post('aggregate')
  public async aggregateModels(
    @Body() modelIds: string[],
    @Res() response: Response,
  ) {
    try {
      console.log(
        '\n--> Submit Transaction: AggregateModels, aggregates list of given models',
      );
      const jsonModelIds = JSON.stringify(modelIds);
      const contract = await this.gatewayService.getContract();
      await contract.submitTransaction('AggregateModels', jsonModelIds);
      console.log('*** Transaction committed successfully');
      response
        .status(HttpStatus.OK)
        .json({ message: 'Models aggregated succesfully' });
    } catch (error: any) {
      console.error('Error aggregating models', error.message);
      response
        .status(HttpStatus.BAD_REQUEST)
        .json({ message: 'Error aggregating models', error: error.message });
    }
  }

  /**
   * TEMP CODE --------------------------------------------------------
   */

  // temp code to remove
  // used for debugging purposes
  // this code goes into the chaincode
  @Post('aggregateStub')
  public async aggregateModelsStub(
    @Body() modelIds: string[],
    @Res() response: Response,
  ) {
    const contract = await this.gatewayService.getContract();
    const aggregatedWeights: ModelParams = {};

    const modelWeights: ModelParams[] = [];
    // get all the models from the modelIds array
    for (const modelId of modelIds) {
      const resultBytes = await contract.evaluateTransaction(
        'ReadModel',
        modelId,
      );
      const resultJson = GatewayService.UTF8_DECODER.decode(resultBytes);
      const modelJSON = JSON.parse(resultJson);
      if (!modelJSON || modelJSON.length === 0) {
        throw new Error(`The model ${modelId} does not exist`);
      }
      const modelParams: ModelParams = await this.deserializeModelParams(
        modelJSON.modelParams,
      );
      // fill up modelWeights with all models' weights
      modelWeights.push(modelParams);
    }
    // aggregate the weights
    for (const key in modelWeights[0]) {
      aggregatedWeights[key] = this.aggregateWeights(
        modelWeights.map((model) => model[key]),
      );
    }
    // serialize the aggregated weights
    const encodedAggregatedParams =
      await this.serializeModelParams(aggregatedWeights);
    // save it as a new model using this.CreateModel
    const newModelId = modelIds[0] + modelIds[1];
    // Save the new aggregated model
    await contract.submitTransaction(
      'CreateModel',
      newModelId,
      encodedAggregatedParams,
      'Victor',
    );
    // return the new model
    const resultBytes = await contract.evaluateTransaction(
      'ReadModel',
      newModelId,
    );
    // await this.CreateModel(ctx, newModelId, encodedAggregatedParams, 'Victor');

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
            modelWeights.map((array) => (array as ModelWeights[])[i]),
          ),
        );
      return aggregated;
    }
  }

  private async deserializeModelParams(
    encodedModelParams: string,
  ): Promise<ModelParams> {
    // 1. Decode from base64
    const decodedData = Buffer.from(encodedModelParams, 'base64');
    // 2. Decompress from zlib
    const decompressedData = await new Promise<Buffer>((resolve, reject) => {
      zlib.unzip(decodedData, (err, buffer) => {
        if (err) reject(err);
        else resolve(buffer);
      });
    });
    // 3. Deserialize using MessagePack
    const unpackedData: ModelParams = decode(decompressedData) as ModelParams;
    return unpackedData;
  }

  private async serializeModelParams(
    modelParams: ModelParams,
  ): Promise<string> {
    // 1. Serialize using MessagePack
    const packedData = encode(modelParams);
    // 2. Compress using zlib
    const compressedData = await new Promise<Buffer>((resolve, reject) => {
      zlib.deflate(packedData, (err, buffer) => {
        if (err) reject(err);
        else resolve(buffer);
      });
    });
    // 3. Encode to base64
    const encodedData = compressedData.toString('base64');
    return encodedData;
  }
  private isArrayOfNumbers(array: ModelWeights[]): array is number[] {
    return array.every((element) => typeof element === 'number');
  }
}
