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

  constructor(private sanitizer: DomSanitizer,private service: AimodelService) {}

  async onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      this.fileUploadSubscription = await this.service.onFileUpload(file).subscribe(response=>{
        const {message} = response;

        if (message == "Video processing completed"){
          this.completionSub = this.service.downloadFile().subscribe(responsesub=>{
            if (responsesub.size === 0) {
              console.error('Received empty Blob. Check API response.');
              return;
            }

            const objectURL = URL.createObjectURL(responsesub); // Convert Blob to URL
            this.videoUrl = this.sanitizer.bypassSecurityTrustUrl(objectURL);
          });
        }
      });

    }
  }

  ngOnDestroy(): void {
   this.fileUploadSubscription.unsubscribe();
   this.completionSub.unsubscribe();
  }
}



