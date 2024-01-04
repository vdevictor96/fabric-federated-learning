import { Global, MiddlewareConsumer, Module, NestModule } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { GatewayService } from './gateway.service';
import { GatewayController } from './gateway.controller';
import { GzipMiddleware } from './middleware/gzip.middleware';

@Global()
@Module({
  imports: [],
  controllers: [AppController, GatewayController],
  providers: [AppService, GatewayService],
})
export class AppModule implements NestModule {
  configure(consumer: MiddlewareConsumer) {
    consumer.apply(GzipMiddleware).forRoutes('*compressed*');
  }
}
