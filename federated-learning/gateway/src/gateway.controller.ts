import {
  Controller,
  Get,
  HttpStatus,
  OnModuleDestroy,
  OnModuleInit,
  Post,
  Res,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { GatewayService } from './gateway.service';
import { FileInterceptor } from '@nestjs/platform-express';
import { promises as fs } from 'fs';
import * as path from 'path';
import { Contract } from '@hyperledger/fabric-gateway';
import { Response } from 'express';

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
        '\n--> Submit Transaction: InitLedger, function creates the initial set of assets on the ledger',
      );
      const contract = await this.gatewayService.getContract();
      const res = await contract.submitTransaction('InitLedger');
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
}
