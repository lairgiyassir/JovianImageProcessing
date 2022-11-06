import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { map, Observable, tap } from 'rxjs';
import { ImageId } from '../shared/image-id';
import { ImageResponse } from '../shared/image-response';

@Injectable({
  providedIn: 'root'
})
export class SearchService {

  baseApiUrl = "http://localhost:8000/process"

  constructor(private http : HttpClient) { }


  sendId(id:ImageId) : Observable<ImageResponse> {

    return this.http.post<ImageResponse>(this.baseApiUrl,id).pipe(
      tap(e => console.log(e))
    );
  }


}
