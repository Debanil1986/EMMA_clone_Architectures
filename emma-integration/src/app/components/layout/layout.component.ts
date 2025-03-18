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
  videoUrl: SafeUrl | null = null;
  fileUploadSubscription= new Subscription();
  completionSub: Subscription = new Subscription();
  errorMessage:string="";
  objectUrl!: string;

  constructor(private sanitizer: DomSanitizer,private service: AimodelService) {}




  private convertBlobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const dataUrl = reader.result as string;
        // Extract Base64 portion from Data URL
        const base64 = dataUrl.split(',')[1];
        resolve(`data:video/mp4;base64,${base64}`);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

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
                this.objectUrl = await this.convertBlobToBase64(responsesub);
                console.log('this.objectUrl: ', this.objectUrl);
                }catch(err){console.error(err)};
                this.videoUrl = this.sanitizer.bypassSecurityTrustUrl(this.objectUrl);

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
    URL.revokeObjectURL(this.objectUrl);
  }
  }
}



