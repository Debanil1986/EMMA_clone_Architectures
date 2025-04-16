import { AimodelService } from './../../services/aimodel.service';
import { CommonModule } from '@angular/common';
import { HttpClient, provideHttpClient, withFetch } from '@angular/common/http';
import { Component, OnDestroy } from '@angular/core';
import { bootstrapApplication, DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-layout',
  standalone: true,
  imports: [
    CommonModule],
  templateUrl: './layout.component.html',
  styleUrl: './layout.component.css'
})
export class LayoutComponent implements OnDestroy {
  isDragging = false;
  videoUrl: SafeUrl | any;
  fileUploadSubscription= new Subscription();
  completionSub: Subscription = new Subscription();
  errorMessage:string="";
  objectUrl!: string;

  constructor(private sanitizer: DomSanitizer,private service: AimodelService) {}





  async onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      this.fileUploadSubscription = await this.service.onFileUpload(file).subscribe(response=>{
        const {message} = response;

        if (message == "Video processing completed"){
          this.completionSub = this.service.downloadFile().subscribe({
            next:async (responsesub)=>{
                console.log(responsesub)
                try{
                  const webmBlob = new Blob([responsesub], { type: 'video/mp4' })
                const blobconvert =URL.createObjectURL(webmBlob);
                this.videoUrl = this.sanitizer.bypassSecurityTrustUrl(blobconvert);
                console.log(JSON.stringify(this.videoUrl));
              }catch(err){console.error(err)};

          },error:(error)=>{
              this.errorMessage =error
              console.error('fetching video problem:', error);
          }
    });
        }
      });

    }
  }

  ngOnDestroy(): void {
   this.fileUploadSubscription.unsubscribe();
   this.completionSub.unsubscribe();
   if (this.objectUrl) {
    URL.revokeObjectURL(this.videoUrl.toString());
  }
  }
}



