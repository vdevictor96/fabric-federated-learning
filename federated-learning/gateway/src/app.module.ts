import { Global, Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { GatewayService } from './gateway.service';
import { GatewayController } from './gateway.controller';

@Global()
@Module({
  imports: [],
  controllers: [AppController, GatewayController],
  providers: [AppService, GatewayService],
})
export class AppModule {}
