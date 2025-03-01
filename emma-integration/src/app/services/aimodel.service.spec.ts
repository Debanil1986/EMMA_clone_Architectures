import { TestBed } from '@angular/core/testing';

import { AimodelService } from './aimodel.service';

describe('AimodelService', () => {
  let service: AimodelService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AimodelService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
