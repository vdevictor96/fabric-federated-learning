import { Injectable, NestMiddleware } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';
import * as zlib from 'zlib';

@Injectable()
export class GzipMiddleware implements NestMiddleware {
  use(req: Request, res: Response, next: NextFunction) {
    //   console.log('Request...');
    //   next();
    // }
    if (req.headers['content-encoding'] === 'gzip') {
      const gunzip = zlib.createGunzip();
      req
        .pipe(gunzip)
        .on('data', (chunk) => {
          if (!req.body) {
            req.body = '';
          }
          req.body += chunk.toString();
        })
        .on('end', () => {
          next();
        })
        .on('error', (err) => {
          next(err);
        });
    } else {
      next();
    }
  }
}
