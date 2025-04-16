import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LayoutComponent } from './layout.component';
import { By, DomSanitizer } from '@angular/platform-browser';
import { HttpClientTestingModule, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
declare const jest: any;

describe('LayoutComponent', () => {
  let component: LayoutComponent;
  let fixture: ComponentFixture<LayoutComponent>;


  beforeEach(async () => {
    await TestBed.configureTestingModule({
      providers: [
        provideHttpClient(), // Provide the HttpClient along with HttpClientTesting
        provideHttpClientTesting(),
      ],
      imports: [
        LayoutComponent,

      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LayoutComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('onFileSelected method test', async () => {
    // Mock event object with a valid video file
    const file =new File([''], 'test.mp4', { type: 'video/mp4' })
    const event = {
        target: {
            files: [file]
        }
    };

    // Mock service methods
    const mockService = {
    onFileUpload: jasmine.createSpy().and.returnValue(Promise.resolve({ message: 'Video processing completed' })),
    downloadFile: jasmine.createSpy().and.returnValue(Promise.resolve(new Uint8Array(10)))
    };
    // Mock sanitizer
    const mockSanitizer = jasmine.createSpyObj('DomSanitizer', ['bypassSecurityTrustUrl']);
    const url = 'blob:http://localhost:4200/3e8fa174-fa9a-468e-b4ef-870a04309825';
    mockSanitizer.bypassSecurityTrustUrl.and.callFake(() => `Mocked trusted URL`);


    // Create instance of the class or component containing onFileSelected method
    const instance = component;

    // Call the method with the mock event

    const fakeTrustedUrl = mockSanitizer.bypassSecurityTrustUrl(url);


    const apiData = {
      responseData: 'valid', // Example of API data that determines the operation
    };




    // Call onFileSelected once with the mocked event
    await component.onFileSelected(event);

    const componentHtml = fixture.debugElement.nativeElement.outerHTML;
    console.log(componentHtml);

    const videoElement = fixture.debugElement.query(By.css('.upload-container input[type="file"]'));

    if (videoElement) {
    // Expectations
    const videoNative = videoElement.nativeElement;
    videoNative.click();
    expect(mockService.onFileUpload).toHaveBeenCalled();
    expect(mockService.downloadFile).toHaveBeenCalled();
    }else {
      fail('Video element not found in the component template');
    }

    expect(fakeTrustedUrl).toBe('Mocked trusted URL');
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
