import { NestFactory } from '@nestjs/core';
import type { NestExpressApplication } from '@nestjs/platform-express';

import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create<NestExpressApplication>(AppModule, {
    rawBody: true,
  });
  app.useBodyParser('text', { limit: '50mb', type: 'text/plain' });
  app.useBodyParser('json', { limit: '500mb' });
  app.useBodyParser('raw', { limit: '50mb' });
  app.useBodyParser('urlencoded', { limit: '50mb', extended: true });
  await app.listen(3000);
}
bootstrap();
