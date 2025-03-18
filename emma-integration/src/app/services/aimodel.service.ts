import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { map, Observable,of } from 'rxjs';
import { ResponseData } from '../models/response.model';

@Injectable({
  providedIn: 'root'
})
export class AimodelService {

  constructor(private http: HttpClient) { }

  onFileUpload(videoFile:File):Observable<ResponseData>{
    const formData = new FormData();
    formData.append('title', 'Your Title');
    formData.append('video', videoFile);

    return this.http.post<ResponseData>("http://127.0.0.1:3000/convert-video-to-base64", formData);
  }

  downloadFile():Observable<any>{
    return this.http.get("http://127.0.0.1:3000/download",{ observe:'body',responseType: 'blob' });
  }
}
