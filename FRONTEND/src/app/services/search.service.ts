import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { map, Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SearchService {

  baseApiUrl = "http://localhost:8000/process"

  constructor(private http : HttpClient) { }


  sendId(id:String) : Observable<Object> {

    return this.http.get<Object>(this.baseApiUrl+"/"+id).pipe(
      map(result => {return result})
    );
  }


}
