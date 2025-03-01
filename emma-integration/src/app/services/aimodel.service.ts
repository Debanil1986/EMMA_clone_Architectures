import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable,of } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AimodelService {

  constructor(private http: HttpClient) { }

  onFileUpload(videoFile:File):Observable<any>{
    const formData = new FormData();
    formData.append('title', 'Your Title');
    formData.append('video', videoFile);

    return this.http.post('http://127.0.0.1:3000/convert-video-to-base64', formData);
  }
}
