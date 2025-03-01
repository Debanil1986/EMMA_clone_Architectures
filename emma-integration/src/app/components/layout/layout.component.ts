import { AimodelService } from './../../services/aimodel.service';
import { CommonModule } from '@angular/common';
import { HttpClient, provideHttpClient, withFetch } from '@angular/common/http';
import { Component } from '@angular/core';
import { bootstrapApplication, DomSanitizer, SafeUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-layout',
  standalone: true,
  imports: [
    CommonModule],
  templateUrl: './layout.component.html',
  styleUrl: './layout.component.css'
})
export class LayoutComponent {
  isDragging = false;
  videoUrl: SafeUrl | null = null;

  constructor(private sanitizer: DomSanitizer,private service: AimodelService) {}

  async onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      const result = await this.service.onFileUpload(file).subscribe(response=>{
        console.log(response);
      })
      // this.videoUrl = this.sanitizer.bypassSecurityTrustUrl(
      //   URL.createObjectURL(file)
      // );
    }
  }
}



