import {
  Body,
  Controller,
  Get,
  HttpStatus,
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
import { ModelDto } from './model.dto'; // Adjust the import path as needed

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
    return response
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
      return response
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
      return response
        .status(400)
        .json({ message: 'Error retreiving models', error: error.message });
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
        '\n--> Submit Transaction: CreateModel, creates new model with ID, Size, Owner',
      );
      const contract = await this.gatewayService.getContract();
      await contract.submitTransaction(
        'CreateModel',
        modelDto.id,
        modelDto.size.toString(),
        modelDto.owner,
      );
      console.log('*** Transaction committed successfully');
      return response
        .status(HttpStatus.OK)
        .json({ message: 'Model created succesfully' });
    } catch (error: any) {
      console.error('Error creating model', error.message);
      return response
        .status(HttpStatus.BAD_REQUEST)
        .json({ message: 'Error creating model', error: error.message });
    }
  }
}