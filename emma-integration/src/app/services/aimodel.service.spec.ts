import { TestBed } from '@angular/core/testing';

import { AimodelService } from './aimodel.service';
import { HttpClientTestingModule } from '@angular/common/http/testing';

describe('AimodelService', () => {
  let service: AimodelService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [AimodelService]
    });
    service = TestBed.inject(AimodelService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });


});
